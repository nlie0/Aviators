import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import mne
from sklearn.metrics import confusion_matrix, accuracy_score
from brainda.algorithms.utils.model_selection import (
    set_random_seeds, 
    generate_loo_indices, match_loo_indices)
from brainda.algorithms.decomposition import (
    FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP,
    generate_filterbank, generate_cca_references)
from collections import OrderedDict
from sklearn.pipeline import clone
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import sys, time
from matplotlib.colors import LogNorm
import os, pickle
import argparse

parser = argparse.ArgumentParser(description='Train TRCA model')

# folder path, model name, class count
N_CLASSES   = 5
N_PER_CLASS = 2
STIM_DURATION = 1.2
folder_path = f'data/cyton8_rocket-vep_{N_CLASSES}-class_{STIM_DURATION}s/'
model_save_dir = 'cache/'
model_name = 'FBTRCA_rocket_model.pkl'
sampling_rate = 250

# load from run-N/ subfolders instead of flat run files
run_dirs = sorted([
    os.path.join(folder_path, d)
    for d in os.listdir(folder_path)
    if d.startswith('run-') and os.path.isdir(os.path.join(folder_path, d))
])

reverted_eeg_trials_list = []

for run_dir in run_dirs:
    run_number = int(os.path.basename(run_dir).split('-')[1])

    eeg_file = os.path.join(run_dir, 'eeg_trials.npy')
    eeg_trials = np.load(eeg_file, allow_pickle=True)

    # handle object array with variable-length trials + force float64
    if eeg_trials.dtype == object:
        min_samples = min(t.shape[-1] for t in eeg_trials)
        eeg_trials = np.stack([t[..., :min_samples] for t in eeg_trials]).astype(np.float64)
    else:
        eeg_trials = eeg_trials.astype(np.float64)

    np.random.seed(run_number)
    shuffled_indices = np.random.permutation(eeg_trials.shape[0])
    reverted_eeg_trials = np.empty_like(eeg_trials)
    reverted_eeg_trials[shuffled_indices] = eeg_trials

    # reshape to (N_PER_CLASS, N_CLASSES, 8, n_samples)
    reverted_eeg_trials = reverted_eeg_trials.reshape(N_PER_CLASS, N_CLASSES, 8, -1)
    reverted_eeg_trials_list.append(reverted_eeg_trials)

# Ensure all runs have same sample count before concatenating
min_run_samples = min(r.shape[-1] for r in reverted_eeg_trials_list)
reverted_eeg_trials_list = [r[..., :min_run_samples] for r in reverted_eeg_trials_list]

combined_eeg_trials = np.concatenate(reverted_eeg_trials_list, axis=0)
print("Combined shape:", combined_eeg_trials.shape)

def run_fbtrca(eeg, target_by_trial, target_tab, duration=1, onset_delay=42,srate=300, ensamble=True, return_prob=False,return_template_xcorr=False, return_matching_xcorr=False, print_acc=False):
    eeg = np.copy(eeg)
    np.random.seed(64)
    np.random.shuffle(eeg)
    n_trials = eeg.shape[0]
    classes = range(N_CLASSES)  # was range(32), but now 5 for lanes
    n_classes = len(classes)
    y = np.array([list(target_tab.values())] * n_trials).T.reshape(-1)
    eeg_temp = eeg[:n_trials,classes,:,onset_delay:]
    X = eeg_temp.swapaxes(0,1).reshape(-1,*eeg_temp.shape[2:])
    X = X.astype(np.float64)  # CHANGED: force float64 to avoid dtype('O') error in brainda

    n_bands = 3
    wp = [[8*i, 90] for i in range(1, n_bands+1)]
    ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
    filterbank = generate_filterbank(
        wp, ws, srate, order=4, rp=1)
    filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25
    set_random_seeds(64)
    l = 5
    models = OrderedDict([
        ('fbtrca', FBTRCA(
            filterbank, filterweights=filterweights,ensemble=ensamble)),
    ])
    events = []
    for j_class in classes:
        events.extend([str(target_by_trial[i_trial][j_class]) for i_trial in range(n_trials)])
    events = np.array(events)
    subjects = ['1'] * (n_classes*n_trials)
    meta = pd.DataFrame(data=np.array([subjects,events]).T, columns=["subject", "event"])
    set_random_seeds(42)
    loo_indices = generate_loo_indices(meta)

    for model_name in models:
        if model_name == 'fbtdca':
            filterX, filterY = np.copy(X[..., :int(srate*duration)+l]), np.copy(y)
        else:
            filterX, filterY = np.copy(X[..., :int(srate*duration)]), np.copy(y)
        
        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
        filterX = filterX.astype(np.float64)  # CHANGED: ensure float64 after DC removal

        n_loo = len(loo_indices['1'][events[0]])
        prob_matrices=np.zeros((n_loo,n_classes,n_classes))
        txcorr_matrices=np.zeros((n_loo,n_classes,n_classes))
        mxcorr_matrices=np.zeros((n_loo,n_classes,n_classes))
        loo_accs = []
        testYs = []
        pred_labelss = []
        for k in range(n_loo):
            train_ind, validate_ind, test_ind = match_loo_indices(
                k, meta, loo_indices)
            train_ind = np.concatenate([train_ind, validate_ind])

            trainX, trainY = filterX[train_ind], filterY[train_ind]
            testX, testY = filterX[test_ind], filterY[test_ind]

            model = clone(models[model_name]).fit(
                trainX, trainY,
            )
            if return_template_xcorr:
                templates = np.copy(model.estimators_[0].templates_)
                U = np.copy(model.estimators_[0].Us_)[:, :, :model.n_components]
                U = np.concatenate(U, axis=-1)
                new_templates=np.zeros((templates.shape[0],templates.shape[0]*templates.shape[2]))
                for i_template, template in enumerate(templates):
                    new_templates[i_template] = np.reshape((U.T@template),(-1))
                templates = np.copy(new_templates)
                for i in range(n_classes):
                    for j in range(n_classes):
                        a1 = templates[i]
                        a2 = templates[j]
                        txcorr_matrices[k,i,j] = pearsonr(a1,a2)[0]
            if return_matching_xcorr:
                templates = np.copy(model.estimators_[0].templates_)
                U = np.copy(model.estimators_[0].Us_)[:, :, :model.n_components]
                U = np.concatenate(U, axis=-1)
                eegX = model.transform_filterbank(np.copy(testX))[0]
                new_templates=np.zeros((templates.shape[0],templates.shape[0]*templates.shape[2]))
                new_eegX = np.zeros((eegX.shape[0],eegX.shape[0]*eegX.shape[2]))
                for i_template, template in enumerate(templates):
                    new_templates[i_template] = np.reshape((U.T@template),(-1))
                templates = np.copy(new_templates)
                for i_x, x in enumerate(eegX):
                    new_eegX[i_x] = np.reshape((U.T@x),(-1))
                eegX = np.copy(new_eegX)
                for i in range(n_classes):
                    for j in range(n_classes):
                        a1 = templates[i]
                        a2 = eegX[j]
                        mxcorr_matrices[k,i,j] = pearsonr(a1,a2)[0]
            if return_prob:
                prob_matrices[k] = model.transform(testX)
            pred_labels = model.predict(testX)
            loo_accs.append(
                balanced_accuracy_score(testY, pred_labels))
            pred_labelss.extend(pred_labels)
            testYs.extend(testY)
    if print_acc:
        print("Model:{} LOO Acc:{:.2f}".format(model_name, np.mean(loo_accs)))
    if return_template_xcorr:
        return txcorr_matrices, accuracy_score(testYs, pred_labelss), model
    if return_matching_xcorr:
        return mxcorr_matrices, accuracy_score(testYs, pred_labelss), model
    if return_prob:
        return prob_matrices, accuracy_score(testYs, pred_labelss)
    return confusion_matrix(testYs, pred_labelss, normalize='true'), accuracy_score(testYs, pred_labelss), model

def run_fbtdca(eeg, target_by_trial, target_tab, duration=1.0, onset_delay=42,srate=300, return_prob=True):
    eeg = np.copy(eeg)
    np.random.seed(64)
    np.random.shuffle(eeg)
    n_trials = eeg.shape[0]
    classes = range(N_CLASSES)  # CHANGED: was range(32)
    n_classes = len(classes)
    prob_matrix=np.zeros((n_classes,n_classes))
    y = np.array([list(target_tab.values())] * n_trials).T.reshape(-1)
    eeg_temp = eeg[:n_trials,classes,:,onset_delay:]
    X = eeg_temp.swapaxes(0,1).reshape(-1,*eeg_temp.shape[2:])


    freq_targets = np.array(target_by_trial)[0,:,0]
    phase_targets = np.array(target_by_trial)[0,:,1]
    n_harmonics = 5
    n_bands = 3
    Yf = generate_cca_references(
        freq_targets, srate, duration, 
        phases=phase_targets, 
        n_harmonics=n_harmonics)
    wp = [[8*i, 90] for i in range(1, n_bands+1)]
    ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
    filterbank = generate_filterbank(
        wp, ws, srate, order=4, rp=1)
    filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25
    set_random_seeds(64)
    l = 5
    models = OrderedDict([
        ('fbtdca', FBTDCA(
                filterbank, l, n_components=8, 
                filterweights=filterweights)),
    ])
    events = []
    for j_class in classes:
        events.extend([str(target_by_trial[i_trial][j_class]) for i_trial in range(n_trials)])
    events = np.array(events)
    subjects = ['1'] * (n_classes*n_trials)
    meta = pd.DataFrame(data=np.array([subjects,events]).T, columns=["subject", "event"])
    set_random_seeds(42)
    loo_indices = generate_loo_indices(meta)

    for model_name in models:
        if model_name == 'fbtdca':
            filterX, filterY = np.copy(X[..., :int(srate*duration)+l]), np.copy(y)
        else:
            filterX, filterY = np.copy(X[..., :int(srate*duration)]), np.copy(y)
        
        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

        n_loo = len(loo_indices['1'][events[0]])
        loo_accs = []
        testYs = []
        pred_labelss = []
        for k in range(n_loo):
            train_ind, validate_ind, test_ind = match_loo_indices(
                k, meta, loo_indices)
            train_ind = np.concatenate([train_ind, validate_ind])

            trainX, trainY = filterX[train_ind], filterY[train_ind]
            testX, testY = filterX[test_ind], filterY[test_ind]

            model = clone(models[model_name]).fit(
                trainX, trainY,
                Yf=Yf
            )
            if return_prob:
                prob_matrix+=model.transform(testX)
            pred_labels = model.predict(testX)
            loo_accs.append(
                balanced_accuracy_score(testY, pred_labels))
            pred_labelss.extend(pred_labels)
            testYs.extend(testY)
        
    if return_prob:
        return prob_matrix, accuracy_score(testYs, pred_labelss)
    return confusion_matrix(testYs, pred_labelss, normalize='true'), accuracy_score(testYs, pred_labelss)

# 5 rocket stimulus classes instead of 32
stimulus_classes = [(8, 0), (10, 0), (12, 0), (14, 0), (15, 0)]

target_tab = {tuple(map(float, cls)): idx for idx, cls in enumerate(stimulus_classes)}
target_by_trial = [stimulus_classes] * 99

sampling_rate = 250
baseline_duration = 0.2
baseline_samples = int(baseline_duration * sampling_rate)

baseline_average = np.mean(combined_eeg_trials[:, :, :, :baseline_samples], axis=3, keepdims=True)
baseline_corrected_eeg_trials = combined_eeg_trials - baseline_average
cropped_eeg_trials = combined_eeg_trials[:, :, :, baseline_samples:]

cm, acc, model = run_fbtrca(cropped_eeg_trials, target_by_trial, target_tab, duration=1.2, onset_delay=0, ensamble=True, print_acc=True, srate=250)

os.makedirs(model_save_dir, exist_ok=True)
with open(model_save_dir + model_name, 'wb') as f:
    pickle.dump(model, f)