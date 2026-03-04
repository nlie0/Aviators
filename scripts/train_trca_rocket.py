import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import mne
from sklearn.metrics import confusion_matrix, accuracy_score
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices,
    match_loo_indices,
)
from brainda.algorithms.decomposition import (
    FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP,
    generate_filterbank, generate_cca_references
)
from collections import OrderedDict
from sklearn.pipeline import clone
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import sys, time
from matplotlib.colors import LogNorm
import os, pickle
import argparse

# same STIMULUS_CLASSES in run_rocket.py
STIMULUS_CLASSES = [
    (8, 0),
    (10, 0),
    (12, 0),
    (14, 0),
    (15, 0),
]
N_CLASSES      = len(STIMULUS_CLASSES)  # 5
N_PER_CLASS    = 2                      # must match N_PER_CLASS used during calibration
N_EEG_CHANNELS = 8
SAMPLING_RATE  = 250
STIM_DURATION  = 1.2                    
BASELINE_SEC   = 0.2                    

# Base directory that contains all run-N/ subfolders.
# The trainer will autom find/load every run-N/ folder inside here,
# don't have to change inbetween runs
BASE_DIR = f"data/cyton8_rocket-vep_{N_CLASSES}-class_{STIM_DURATION}s/"

# saves trained model
MODEL_SAVE_DIR = "cache/"
MODEL_NAME     = "FBTRCA_rocket_model.pkl"

BASELINE_SAMPLES = int(BASELINE_SEC * SAMPLING_RATE)
STIM_SAMPLES     = int(STIM_DURATION * SAMPLING_RATE)

# finds all the run-N/ sub folders

if not os.path.isdir(BASE_DIR):
    raise FileNotFoundError(
        f"Base data directory not found: {BASE_DIR}\n"
        "Make sure you have run the game with CALIBRATION_MODE=True, MODE='bci' at least once,\n"
        f"and that you are launching this script from the project root folder."
    )

run_dirs = sorted([
    os.path.join(BASE_DIR, d)
    for d in os.listdir(BASE_DIR)
    if d.startswith("run-") and os.path.isdir(os.path.join(BASE_DIR, d))
])

if len(run_dirs) == 0:
    raise FileNotFoundError(
        f"No run-N/ subfolders found inside {BASE_DIR}.\n"
        "Expected folders named like: run-1/, run-2/, etc."
    )

print(f"Found {len(run_dirs)} run folder(s): {[os.path.basename(d) for d in run_dirs]}")

# Load and reconstruct trial data from every run folder

# Each run saves eeg_trials as a flat object array of length (N_PER_CLASS * N_CLASSES),
# shuffled with seed = run_number.  We un-shuffle and reshape to
# (N_PER_CLASS, N_CLASSES, n_channels, n_samples) before combining.

reverted_list = []

for run_dir in run_dirs:
    # Derive the run number from the folder name (e.g. "run-2" -> 2)
    try:
        run_number = int(os.path.basename(run_dir).split("-")[1])
    except (IndexError, ValueError):
        print(f"  WARNING: Could not parse run number from folder '{run_dir}', skipping.")
        continue

    eeg_file = os.path.join(run_dir, "eeg_trials.npy")
    if not os.path.exists(eeg_file):
        print(f"  WARNING: eeg_trials.npy not found in {run_dir}, skipping.")
        continue

    eeg_trials = np.load(eeg_file, allow_pickle=True)

    # Stack object array -> (N_total, n_channels, n_samples)
    if isinstance(eeg_trials, np.ndarray) and eeg_trials.dtype == object:
        eeg_trials = np.stack(eeg_trials)

    print(f"  Loaded run-{run_number}: raw shape {eeg_trials.shape}")

    n_total  = eeg_trials.shape[0]
    expected = N_PER_CLASS * N_CLASSES
    if n_total != expected:
        print(
            f"  WARNING: expected {expected} trials "
            f"(N_PER_CLASS={N_PER_CLASS} x N_CLASSES={N_CLASSES}), "
            f"got {n_total}. Skipping run-{run_number}."
        )
        continue

    # Un-shuffle using same seed the game used (seed = RUN = run_number)
    np.random.seed(run_number)
    shuffled_indices = np.random.permutation(n_total)
    reverted = np.empty_like(eeg_trials)
    reverted[shuffled_indices] = eeg_trials

    # Reshape to (N_PER_CLASS, N_CLASSES, n_channels, n_samples)
    reverted = reverted.reshape(N_PER_CLASS, N_CLASSES, N_EEG_CHANNELS, -1)
    reverted_list.append(reverted)
    print(f"  run-{run_number} reverted shape: {reverted.shape}")

if len(reverted_list) == 0:
    raise RuntimeError(
        "No valid trial files could be loaded.\n"
        f"Checked folders: {run_dirs}\n"
        "Check your data directory, N_PER_CLASS, and N_CLASSES settings."
    )

# Concatenate all runs along the reps axis -> (total_reps, N_CLASSES, n_channels, n_samples)
# e.g. 2 runs x N_PER_CLASS=2 -> (4, 5, 8, 350)
combined = np.concatenate(reverted_list, axis=0)
total_reps = combined.shape[0]
print(f"Combined shape across {len(reverted_list)} run(s): {combined.shape}  "
      f"({total_reps} reps total = {len(reverted_list)} runs x {N_PER_CLASS} per class)")

# ---------------------------------------------------------------------------
# Baseline correction and cropping
# ---------------------------------------------------------------------------

# Subtract per-trial, per-channel baseline mean
baseline_mean = np.mean(combined[..., :BASELINE_SAMPLES], axis=-1, keepdims=True)
combined -= baseline_mean

# Crop out the baseline period — model only sees the stimulus window
cropped = combined[..., BASELINE_SAMPLES:]   # -> (reps, N_CLASSES, n_channels, stim_samples)
print(f"Cropped (post-baseline) shape: {cropped.shape}") 

# ---------------------------------------------------------------------------
# FBTRCA training
# ---------------------------------------------------------------------------

def run_fbtrca_5class(
    eeg,
    stimulus_classes,
    srate=250,
    duration=1.2,
    n_bands=3,
    ensemble=True,
    print_acc=True,
):
    """
    Train and LOO-evaluate FBTRCA on a 5-class rocket SSVEP dataset.

    Parameters
    ----------
    eeg : np.ndarray, shape (n_reps, n_classes, n_channels, n_samples)
    stimulus_classes : list of (freq, phase) tuples, length n_classes
    srate : int
    duration : float  — stimulus duration in seconds
    n_bands : int     — number of sub-bands for filterbank
    ensemble : bool   — use ensemble TRCA

    Returns
    -------
    cm    : confusion matrix (normalized)
    acc   : LOO accuracy
    model : final fitted FBTRCA model (trained on ALL data)
    """
    eeg = np.copy(eeg)
    n_reps, n_classes, n_channels, n_samples = eeg.shape

    # Build label array matching brainda's expected format
    # y has length n_classes * n_reps, ordered by class
    target_tab = {
        tuple(map(float, cls)): idx for idx, cls in enumerate(stimulus_classes)
    }
    class_labels = [str(tuple(map(float, cls))) for cls in stimulus_classes]

    y = np.array(class_labels * n_reps)  # repeat class labels for each rep

    # X shape expected by brainda: (n_classes * n_reps, n_channels, n_samples)
    # Swap axes: (n_reps, n_classes, ...) -> (n_classes, n_reps, ...) -> flatten first two
    X = eeg.swapaxes(0, 1).reshape(-1, n_channels, n_samples)
    X = X - np.mean(X, axis=-1, keepdims=True)  # remove DC

    # Filterbank: sub-bands at [8,90], [16,90], [24,90] Hz
    wp = [[8 * i, 90] for i in range(1, n_bands + 1)]
    ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
    filterbank = generate_filterbank(wp, ws, srate, order=4, rp=1)
    filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25

    set_random_seeds(64)
    models = OrderedDict([
        ("fbtrca", FBTRCA(filterbank, filterweights=filterweights, ensemble=ensemble)),
    ])

    # Build metadata for LOO cross-validation
    subjects = ["1"] * (n_classes * n_reps)
    events = np.array(class_labels * n_reps)
    meta = pd.DataFrame(
        data=np.array([subjects, events]).T,
        columns=["subject", "event"],
    )
    set_random_seeds(42)
    loo_indices = generate_loo_indices(meta)

    filterX = np.copy(X[..., :int(srate * duration)])
    filterY = np.copy(y)

    n_loo = len(loo_indices["1"][events[0]])
    loo_accs = []
    testYs = []
    pred_labelss = []

    for k in range(n_loo):
        train_ind, validate_ind, test_ind = match_loo_indices(k, meta, loo_indices)
        train_ind = np.concatenate([train_ind, validate_ind])

        trainX, trainY = filterX[train_ind], filterY[train_ind]
        testX, testY = filterX[test_ind], filterY[test_ind]

        fitted_model = clone(models["fbtrca"]).fit(trainX, trainY)
        pred_labels = fitted_model.predict(testX)

        loo_accs.append(balanced_accuracy_score(testY, pred_labels))
        pred_labelss.extend(pred_labels)
        testYs.extend(testY)

    loo_acc = np.mean(loo_accs)
    if print_acc:
        print(f"FBTRCA LOO Accuracy: {loo_acc:.4f} ({loo_acc*100:.1f}%)")

    cm = confusion_matrix(testYs, pred_labelss, normalize="true")

    # --- Retrain on ALL data for the final saved model ---
    print("Retraining on full dataset for saved model...")
    final_model = clone(models["fbtrca"]).fit(filterX, filterY)

    return cm, accuracy_score(testYs, pred_labelss), final_model


cm, acc, model = run_fbtrca_5class(
    cropped,
    STIMULUS_CLASSES,
    srate=SAMPLING_RATE,
    duration=STIM_DURATION,
    ensemble=True,
    print_acc=True,
)

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to: {model_path}")
print(f"Point MODEL_PATH in run_rocket.py to: {model_path}")

# ---------------------------------------------------------------------------
# Plot confusion matrix
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
plt.colorbar(im, ax=ax)
class_names = [f"{f}Hz φ{p}" for f, p in STIMULUS_CLASSES]
ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"FBTRCA 5-class LOO Confusion Matrix\nAccuracy: {acc*100:.1f}%")
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, "rocket_confusion_matrix.png"), dpi=150)
print(f"Confusion matrix saved to: {MODEL_SAVE_DIR}rocket_confusion_matrix.png")
plt.show()