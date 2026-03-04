from psychopy import visual, core, event
from psychopy.hardware import keyboard
import numpy as np
import os, time, random, pickle
from scipy import signal
import mne

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MODE = "testing"        # "testing" (keyboard) or "bci" (EEG hardware)
CYTON_IN = True         # only used when MODE="bci"
LSL_OUT = False

CALIBRATION_MODE = False  # True = collect labelled data; False = play with model
N_PER_CLASS = 2
RUN = 1   #For callibration true, change for run 1 and 2, 3 if want more accuracy/first 2 was wack

LANES = 5
WIDTH = 1536
HEIGHT = 864
REFRESH_RATE = 60.02
STIM_DURATION = 1.2
BASELINE_SEC = 0.2
PRE_STIM_PAUSE = 0.6

FULLSCREEN = True
SCREEN_INDEX = 0

# Must match MODEL_NAME in train_trca_rocket.py
MODEL_PATH = "cache/FBTRCA_rocket_model.pkl"

# Save directory stuff and should match FOLDER_PATH in train_trca_rocket.py
SAVE_DIR = f"data/cyton8_rocket-vep_{LANES}-class_{STIM_DURATION}s/run-{RUN}/"
SAVE_FILE_EEG        = os.path.join(SAVE_DIR, "eeg.npy")
SAVE_FILE_AUX        = os.path.join(SAVE_DIR, "aux.npy")
SAVE_FILE_EEG_TRIALS = os.path.join(SAVE_DIR, "eeg_trials.npy")
SAVE_FILE_AUX_TRIALS = os.path.join(SAVE_DIR, "aux_trials.npy")
SAVE_FILE_LABELS     = os.path.join(SAVE_DIR, "labels.npy")

# 5 SSVEP frequencies: one per lane.
STIMULUS_CLASSES = [(8, 0), (10, 0), (12, 0), (14, 0), (15, 0)]

L_FREQ = 2.0
H_FREQ = 40.0

#helper methods
def esc_pressed():
    return "escape" in event.getKeys(keyList=["escape"])


def quit_clean(win, board=None, stop_event=None):
    try:
        if stop_event is not None:
            stop_event.set()
        if board is not None:
            board.stop_stream()
            board.release_session()
    except Exception:
        pass
    try:
        win.close()
    except Exception:
        pass
    core.quit()


def save_calibration_data():
    """Call this whenever the run ends (normally or via escape)."""
    if CALIBRATION_MODE and MODE == "bci":
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(SAVE_FILE_EEG, eeg_buf["eeg"])
        np.save(SAVE_FILE_AUX, eeg_buf["aux"])
        np.save(SAVE_FILE_EEG_TRIALS, np.array(eeg_trials, dtype=object))
        np.save(SAVE_FILE_AUX_TRIALS, np.array(aux_trials, dtype=object))
        np.save(SAVE_FILE_LABELS, np.array(labels))
        print(f"Calibration data saved to {SAVE_DIR}")

#For window display

window = visual.Window(
    size=[WIDTH, HEIGHT],
    fullscr=FULLSCREEN,
    screen=SCREEN_INDEX,
    units="norm",
    allowGUI=False,
    winType="pyglet",
    useRetina=True,
)

w, h = window.size
aspect_ratio = w / h

#scene components

def lane_y_positions():
    return np.linspace(0.65, -0.65, LANES)


Y_LANES = lane_y_positions()


def create_tracks():
    lines = []
    sep_y = np.linspace(0.8, -0.8, LANES + 1)
    for y in sep_y:
        lines.append(
            visual.Line(window, start=(-1, y), end=(1, y),
                        lineColor="white", lineWidth=2)
        )
    return lines


def create_player():
    return visual.Polygon(
        window, edges=3, radius=0.05,
        fillColor="white", lineColor="white",
        pos=(-0.6, Y_LANES[2]), ori=90,
    )


def create_lane_stimuli():
    rects = []
    for y in Y_LANES:
        rects.append(
            visual.Rect(
                window, width=0.18, height=0.18 * aspect_ratio,
                pos=(0.75, y),
                fillColor=[-1, -1, -1],
                lineColor="white", lineWidth=2,
            )
        )
    return rects


def create_photosensor_dot():
    size = 0.06
    return visual.Rect(
        window, units="norm",
        width=size, height=size * aspect_ratio,
        fillColor=[-1, -1, -1],
        lineWidth=0,
        pos=(1 - size / 2, -1 + size * aspect_ratio / 2),
    )


tracks          = create_tracks()
player          = create_player()
lane_rects      = create_lane_stimuli()
photosensor_dot = create_photosensor_dot()

# Game state

PLAYER_X        = -0.6
OBSTACLE_START_X = 1.05
active_obstacles = []
crash_count      = 0
current_lane     = 2


def spawn_obstacles(blocked_lanes):
    global active_obstacles
    active_obstacles = []
    for lane in blocked_lanes:
        rect = visual.Rect(
            window, width=0.10, height=0.14 * aspect_ratio,
            pos=(OBSTACLE_START_X, Y_LANES[lane]),
            fillColor=[1, 0.2, 0.2], lineColor="white", lineWidth=2,
        )
        active_obstacles.append({"lane": lane, "x": OBSTACLE_START_X, "stim": rect})


def update_obstacles(dx):
    for ob in active_obstacles:
        ob["x"] -= dx
        ob["stim"].pos = (ob["x"], Y_LANES[ob["lane"]])


def check_collision(lane):
    for ob in active_obstacles:
        if ob["lane"] == lane and ob["x"] <= PLAYER_X + 0.02:
            return True
    return False


def game_over_screen(board=None, stop_event=None):
    msg = visual.TextStim(
        window,
        text="GAME OVER\n\nPress SPACE to restart\nPress ESC to quit",
        height=0.09, color="white", pos=(0, 0), alignText="center",
    )
    while True:
        keys = event.getKeys()
        if "escape" in keys:
            save_calibration_data()
            quit_clean(window, board, stop_event)
        if "space" in keys:
            return
        window.color = [-1, -1, -1]
        msg.draw()
        window.flip()


def draw_scene(cur_lane, status_text=None):
    window.color = [-1, -1, -1]
    for t in tracks:
        t.draw()
    for r in lane_rects:
        r.draw()
    for ob in active_obstacles:
        ob["stim"].draw()
    player.pos = (PLAYER_X, Y_LANES[cur_lane])
    player.draw()
    if status_text is not None:
        visual.TextStim(
            window, text=status_text,
            pos=(-0.95, 0.92), height=0.06, color="white",
            alignText="left", anchorHoriz="left", anchorVert="top",
        ).draw()

#stimulus frames

def generate_stimulus_frames():
    num_frames = int(round(STIM_DURATION * REFRESH_RATE))
    frame_indices = np.arange(num_frames)
    stim_frames = np.zeros((num_frames, LANES))
    for i, (f, phase) in enumerate(STIMULUS_CLASSES):
        phase += 1e-5   # nudge away from square-wave discontinuity
        stim_frames[:, i] = signal.square(
            2 * np.pi * f * (frame_indices / REFRESH_RATE) + phase * np.pi
        )
    return stim_frames


stim_frames = generate_stimulus_frames()

# Bci setup
board        = None
stop_event   = None
queue_in     = None
sampling_rate = None
model        = None

# Single dict to hold raw accumulated buffers whiched helped global/scoping bug where two functions modified separate copies of eeg/aux/timestamp.
eeg_buf = {
    "eeg":       np.zeros((8, 0)),
    "aux":       np.zeros((3, 0)),
    "timestamp": np.zeros((0,)),
}

eeg_trials  = []
aux_trials  = []
labels      = []
trial_ends  = []
skip_count  = 0

def _drain_queue():
    """Pull all pending packets from the Cyton thread into eeg_buf."""
    if queue_in is None:
        return
    while not queue_in.empty():
        eeg_in, aux_in, ts_in = queue_in.get()
        eeg_buf["eeg"]       = np.concatenate((eeg_buf["eeg"],       eeg_in),  axis=1)
        eeg_buf["aux"]       = np.concatenate((eeg_buf["aux"],       aux_in),  axis=1)
        eeg_buf["timestamp"] = np.concatenate((eeg_buf["timestamp"], ts_in),   axis=0)


if MODE == "bci" and CYTON_IN:
    import glob, sys, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue

    sampling_rate  = 250
    CYTON_BOARD_ID = 0
    BAUD_RATE      = 115200
    ANALOGUE_MODE  = "/2"

    def find_openbci_port():
        if sys.platform.startswith("win"):
            ports = ["COM%s" % (i + 1) for i in range(256)]
        elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
            ports = glob.glob("/dev/ttyUSB*")
        elif sys.platform.startswith("darwin"):
            ports = glob.glob("/dev/cu.usbserial*")
        else:
            raise EnvironmentError("Unsupported OS")

        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b"v")
                line = ""
                time.sleep(2)
                if s.inWaiting():
                    while "$$$" not in line:
                        line += s.read().decode("utf-8", errors="replace")
                    if "OpenBCI" in line:
                        s.close()
                        return port
                s.close()
            except (OSError, serial.SerialException):
                pass
        raise OSError("Cannot find OpenBCI port.")

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    print(board.config_board("/0"))
    print(board.config_board("//"))
    print(board.config_board(ANALOGUE_MODE))
    board.start_stream(45000)

    stop_event = Event()

    def _cyton_reader(q, lsl_out=False):
        while not stop_event.is_set():
            data_in    = board.get_board_data()
            ts_in      = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in     = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in     = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(ts_in) > 0:
                q.put((eeg_in, aux_in, ts_in))
            time.sleep(0.1)

    from queue import Queue
    queue_in = Queue()
    cyton_thread = Thread(target=_cyton_reader, args=(queue_in, LSL_OUT))
    cyton_thread.daemon = True
    cyton_thread.start()

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(
            f"WARNING: No model found at {MODEL_PATH}. "
            "Run train_trca_rocket.py after collecting calibration data."
        )

# Game phases

def obstacle_warning_phase():
    blocked = set(random.sample(range(LANES), 2))
    spawn_obstacles(blocked)
    t0 = core.getTime()
    while core.getTime() - t0 < 0.8:
        if esc_pressed():
            save_calibration_data()
            quit_clean(window, board, stop_event)
        for i, r in enumerate(lane_rects):
            r.fillColor = [1, 1, -1] if i in blocked else [-1, -1, -1]
        draw_scene(current_lane, status_text=f"MODE: {MODE} | Warning")
        window.flip()
    return blocked


def pre_stimulus_phase():
    for _ in range(2):
        for filled in (True, False):
            if esc_pressed():
                save_calibration_data()
                quit_clean(window, board, stop_event)
            for r in lane_rects:
                r.fillColor = [1, 1, 1] if filled else [-1, -1, -1]
            draw_scene(current_lane, status_text="Get ready...")
            window.flip()
            core.wait(0.15)

    t0 = core.getTime()
    while core.getTime() - t0 < PRE_STIM_PAUSE:
        if esc_pressed():
            save_calibration_data()
            quit_clean(window, board, stop_event)
        draw_scene(current_lane, status_text="Focus now")
        window.flip()


def stimulus_phase_testing():
    """Keyboard(NO BCI) mode: press 1-5 to pick a lane."""
    chosen = current_lane
    t0 = core.getTime()
    while core.getTime() - t0 < STIM_DURATION:
        keys = event.getKeys()
        if "escape" in keys:
            quit_clean(window, board, stop_event)
        for r in lane_rects:
            r.fillColor = [1, -1, -1]
        for k in keys:
            if k in ["1", "2", "3", "4", "5"]:
                chosen = int(k) - 1
        draw_scene(current_lane, status_text=f"Press 1-5 (chosen: {chosen + 1})")
        window.flip()
    return chosen


def stimulus_phase_bci_flicker():
    """
    Display SSVEP flicker on all lane boxes for STIM_DURATION seconds.
    The photosensor dot is flashed white so the aux photodiode records
    the trial boundary in hardware — this is what trial segmentation relies on.

    All queue draining happens HERE (single location) to avoid the
    dual-drain scoping bug from the original.
    """
    num_frames = stim_frames.shape[0]
    for i in range(num_frames):
        if esc_pressed():
            save_calibration_data()
            quit_clean(window, board, stop_event)

        # Drain queue here — the only place raw EEG accumulates
        _drain_queue()

        for j, r in enumerate(lane_rects):
            v = stim_frames[i, j]
            r.fillColor = [v, v, v]

        # Photosensor ON during every stimulus frame
        photosensor_dot.fillColor = [1, 1, 1]

        draw_scene(current_lane, status_text="BCI stimulus")
        photosensor_dot.draw()
        window.flip()

    # Turn photosensor OFF after stimulus
    photosensor_dot.fillColor = [-1, -1, -1]
    for r in lane_rects:
        r.fillColor = [-1, -1, -1]
    draw_scene(current_lane)
    photosensor_dot.draw()
    window.flip()


def predict_lane_from_cyton(trial_index):
    global trial_ends

    if model is None:
        # No model yet: fall back to staying in current lane
        print("No model loaded — staying in current lane.")
        return current_lane

    # Wait for the photosensor edge that marks end of this trial
    while True:
        _drain_queue()
        photo_trigger  = (eeg_buf["aux"][1] > 20).astype(int)
        trial_starts_i = np.where(np.diff(photo_trigger) == 1)[0]
        trial_ends     = np.where(np.diff(photo_trigger) == -1)[0]
        if len(trial_ends) > trial_index + skip_count:
            break
        core.wait(0.01)

    baseline_n  = int(BASELINE_SEC * sampling_rate)
    trial_start = max(trial_starts_i[trial_index + skip_count] - baseline_n, 0)
    trial_len   = int(STIM_DURATION * sampling_rate) + baseline_n

    filtered_eeg = mne.filter.filter_data(
        eeg_buf["eeg"], sfreq=sampling_rate,
        l_freq=L_FREQ, h_freq=H_FREQ, verbose=False,
    )

    trial_eeg = np.copy(filtered_eeg[:, trial_start:trial_start + trial_len])
    trial_aux = np.copy(eeg_buf["aux"][:, trial_start:trial_start + trial_len])

    baseline_avg  = np.mean(trial_eeg[:, :baseline_n], axis=1, keepdims=True)
    trial_eeg    -= baseline_avg

    eeg_trials.append(trial_eeg)
    aux_trials.append(trial_aux)

    cropped = trial_eeg[:, baseline_n:]

    # model trained on 5 classes
    pred = model.predict(cropped)[0]
    return int(pred)


def movement_phase(new_lane):
    global current_lane, crash_count

    current_lane = int(new_lane)
    for r in lane_rects:
        r.fillColor = [-1, -1, -1]

    t0     = core.getTime()
    last_t = t0
    speed  = 1.6

    while core.getTime() - t0 < 1.2:
        if esc_pressed():
            save_calibration_data()
            quit_clean(window, board, stop_event)

        now    = core.getTime()
        dt     = now - last_t
        last_t = now

        update_obstacles(dx=speed * dt)

        if check_collision(current_lane):
            crash_count += 1
            draw_scene(current_lane, status_text=f"CRASH! Total: {crash_count}")
            photosensor_dot.draw()
            window.flip()
            core.wait(0.2)
            return True

        draw_scene(current_lane, status_text=f"Lane {current_lane + 1} | crashes: {crash_count}")
        photosensor_dot.draw()
        window.flip()

    return False

# Calibration trial sequence

if CALIBRATION_MODE and MODE == "bci":
    trial_sequence = np.tile(np.arange(LANES), N_PER_CLASS)
    np.random.seed(RUN)
    np.random.shuffle(trial_sequence)
    print(f"Calibration sequence ({len(trial_sequence)} trials): {trial_sequence}")
else:
    trial_sequence = None

# Main loop for bci game
try:
    trial_index = 0

    while True:
        event.clearEvents()

        obstacle_warning_phase()
        pre_stimulus_phase()

        if MODE == "testing":
            target_lane = stimulus_phase_testing()

        else:  # bci
            stimulus_phase_bci_flicker()

            if CALIBRATION_MODE and trial_sequence is not None:
                seq_idx = trial_index % len(trial_sequence)
                target_lane_label = int(trial_sequence[seq_idx])
                labels.append(target_lane_label)
                print(f"Trial {trial_index}: target lane = {target_lane_label}")

            target_lane = predict_lane_from_cyton(trial_index)
            trial_index += 1

        crashed = movement_phase(target_lane)
        if crashed:
            game_over_screen(board, stop_event)
            crash_count  = 0
            current_lane = 2

except KeyboardInterrupt:
    print("Interrupted by user.")

except Exception as e:
    print(f"Unexpected error: {repr(e)}")
    import traceback
    traceback.print_exc()

finally:
    save_calibration_data()
    quit_clean(window, board, stop_event)