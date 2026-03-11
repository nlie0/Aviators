from psychopy import visual, core, event
from psychopy.hardware import keyboard
import numpy as np
from scipy import signal
import random, os, pickle
import mne

#Settings

cyton_in = True  #True if hardware
USE_SYNTHETIC = False  # set True to test w/brainflow synthetic board (no hardware)
lsl_out = False

MODE = "bci" # "testing (keyboard controls)" or "bci"
CALIBRATION_MODE = False
N_PER_CLASS = 2
RUN = 4
LANES = 5


# Window dimension stuff
width = 1536
height = 864
aspect_ratio = width / height

refresh_rate = 60.02
stim_duration = 1.2
baseline_duration = 0.2
baseline_duration_samples = int(baseline_duration * 250)
num_obstacles = 4
PRE_STIM_PAUSE = 0.6
N_EEG_CHANNELS = 8

# save paths use run-N/ subfolder structure to match train_trca_rocket.py
save_dir = f'data/cyton8_rocket-vep_{LANES}-class_{stim_duration}s/run-{RUN}/'
save_file_eeg        = save_dir + 'eeg.npy'
save_file_aux        = save_dir + 'aux.npy'
save_file_eeg_trials = save_dir + 'eeg_trials.npy'
save_file_aux_trials = save_dir + 'aux_trials.npy'
save_file_labels     = save_dir + 'labels.npy'

model_file_path = 'cache/FBTRCA_rocket_model.pkl'  # rocket model name

# 5 rocket stimulus classes instead of 32 in ssvep speller
stimulus_classes = [(8, 0), (10, 0), (12, 0), (14, 0), (15, 0)]

keyboard = keyboard.Keyboard()
window = visual.Window(
    size=[width, height],
    checkTiming=True,
    allowGUI=False,
    fullscr=True,
    useRetina=False,
)


# rocket UI helper metjods

def lane_y_positions():
    return np.linspace(0.65, -0.65, LANES)

Y_LANES = lane_y_positions()

def create_tracks():
    lines = []
    sep_y = np.linspace(0.8, -0.8, LANES + 1)
    for y in sep_y:
        lines.append(visual.Line(window, start=(-1, y), end=(1, y), lineColor="white", lineWidth=2))
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
        rects.append(visual.Rect(
            window, width=0.18, height=0.18 * aspect_ratio,
            pos=(0.75, y), fillColor=[-1, -1, -1],
            lineColor="white", lineWidth=2,
        ))
    return rects

tracks   = create_tracks()
player   = create_player()
lane_rects = create_lane_stimuli()

photosensor_dot = visual.Rect(
    window, units="norm",
    width=0.06, height=0.06 * aspect_ratio,
    fillColor=[-1, -1, -1], lineWidth=0,
    pos=(0.97, -0.95),
)


# obstacle/game state (replaces letter prediction state)
PLAYER_X       = -0.6
OBSTACLE_START_X = 1.05
active_obstacles = []
crash_count    = 0
current_lane   = 2

# Pre-generate calibration sequence at startup
trial_sequence = None
if CALIBRATION_MODE:
    trial_sequence = np.tile(np.arange(LANES), N_PER_CLASS)
    np.random.seed(RUN)
    np.random.shuffle(trial_sequence)

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

def draw_scene(lane, status_text=None, cue_lane=None):
    window.color = [-1, -1, -1]
    for t in tracks:
        t.draw()
    for i, r in enumerate(lane_rects):
        if cue_lane is not None:
            if i == cue_lane:
                r.lineColor = [1, 1, 0]
                r.lineWidth = 5
                r.width  = 0.22
                r.height = 0.18 * aspect_ratio  # keep height unchanged to avoid overlap
            else:
                r.lineColor = "white"
                r.lineWidth = 2
                r.width  = 0.18
                r.height = 0.18 * aspect_ratio
        else:
            r.lineColor = "white"
            r.lineWidth = 2
            r.width  = 0.18
            r.height = 0.18 * aspect_ratio
        r.draw()
    for ob in active_obstacles:
        ob["stim"].draw()
    player.pos = (PLAYER_X, Y_LANES[lane])
    player.draw()
    if status_text is not None:
        visual.TextStim(
            window, text=status_text,
            pos=(-0.95, 0.92), height=0.06, color="white",
            alignText="left", anchorHoriz="left", anchorVert="top",
        ).draw()


# Cyton / BCI setup 
eeg       = np.zeros((8, 0))
aux       = np.zeros((3, 0))
timestamp = np.zeros((0,))
eeg_trials = []
aux_trials = []
labels     = []
trial_ends = []
skip_count = 0
model      = None
queue_in   = None
stop_event = None
board      = None
sampling_rate = 250

if cyton_in:
    import glob, sys, time, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue

    CYTON_BOARD_ID = 0
    BAUD_RATE      = 115200
    ANALOGUE_MODE  = '/2'

    def find_openbci_port():
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        openbci_port = ''
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                line = ''
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    c = ''
                    while '$$$' not in line:
                        c = s.read().decode('utf-8', errors='replace')
                        line += c
                    if 'OpenBCI' in line:
                        openbci_port = port
                s.close()
            except (OSError, serial.SerialException):
                pass
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port.')
        else:
            return openbci_port

    print(BoardShim.get_board_descr(CYTON_BOARD_ID))
    params = BrainFlowInputParams()
    if USE_SYNTHETIC:
        from brainflow.board_shim import BoardIds
        CYTON_BOARD_ID = BoardIds.SYNTHETIC_BOARD
        board = BoardShim(CYTON_BOARD_ID, params)
        print("Using SYNTHETIC board - no hardware required")
    elif CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
        board = BoardShim(CYTON_BOARD_ID, params)
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
        board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    if not USE_SYNTHETIC:
        print(board.config_board('/0'))
        print(board.config_board('//'))
        print(board.config_board(ANALOGUE_MODE))
    board.start_stream(45000)
    stop_event = Event()

    # Check once if analog channels exist (synthetic board has none)
    try:
        _analog_chs = BoardShim.get_analog_channels(CYTON_BOARD_ID)
        _has_analog = True
    except Exception:
        _analog_chs = None
        _has_analog = False

    def get_data(queue_in_local, lsl_out=False):
        while not stop_event.is_set():
            data_in      = board.get_board_data()
            timestamp_in = data_in[BoardShim.get_timestamp_channel(CYTON_BOARD_ID)]
            all_eeg      = data_in[BoardShim.get_eeg_channels(CYTON_BOARD_ID)]
            eeg_in       = all_eeg[:N_EEG_CHANNELS]  # clip to 8ch (synthetic gives 16)
            if _has_analog:
                aux_in = data_in[_analog_chs]
            else:
                aux_in = np.zeros((3, eeg_in.shape[1]))
            if len(timestamp_in) > 0:
                queue_in_local.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)

    queue_in = Queue()
    cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
    cyton_thread.daemon = True
    cyton_thread.start()

    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = None

# Stimulus frames
num_frames    = np.round(stim_duration * refresh_rate).astype(int)
frame_indices = np.arange(num_frames)
stimulus_frames = np.zeros((num_frames, LANES))
for i_class, (flickering_freq, phase_offset) in enumerate(stimulus_classes):
    phase_offset += .00001
    stimulus_frames[:, i_class] = signal.square(
        2 * np.pi * flickering_freq * (frame_indices / refresh_rate) + phase_offset * np.pi)

# trial sequence over LANES instead of 32
trial_sequence = np.tile(np.arange(LANES), N_PER_CLASS)
np.random.seed(RUN)
np.random.shuffle(trial_sequence)


# Helper methods between calibration and game modes

def quit_clean():
    try:
        if stop_event is not None:
            stop_event.set()
        if board is not None:
            board.stop_stream()
            board.release_session()
    except Exception:
        pass
    core.quit()

def save_data():
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_file_eeg,        eeg)
    np.save(save_file_aux,        aux)

    eeg_arr = np.empty(len(eeg_trials), dtype=object)
    for i, t in enumerate(eeg_trials):
        eeg_arr[i] = t
    aux_arr = np.empty(len(aux_trials), dtype=object)
    for i, t in enumerate(aux_trials):
        aux_arr[i] = t

    np.save(save_file_eeg_trials, eeg_arr)
    np.save(save_file_aux_trials, aux_arr)
    np.save(save_file_labels,     np.array(labels))

def collect_trial_eeg(i_trial):
    """Identical EEG collection logic from original, adapted for rocket globals."""
    global eeg, aux, timestamp, trial_ends

    if not cyton_in or USE_SYNTHETIC:
        # mock trial for no-hardware testing
        total_n  = baseline_duration_samples + int(stim_duration * sampling_rate)
        fake_eeg = np.random.randn(N_EEG_CHANNELS, total_n) * 10
        fake_aux = np.zeros((3, total_n))
        eeg_trials.append(fake_eeg)
        aux_trials.append(fake_aux)
        return None

    while len(trial_ends) <= i_trial + skip_count:
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get()
            print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
            eeg       = np.concatenate((eeg, eeg_in),             axis=1)
            aux       = np.concatenate((aux, aux_in),             axis=1)
            timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
        photo_trigger = (aux[1] > 20).astype(int)
        trial_starts  = np.where(np.diff(photo_trigger) == 1)[0]
        trial_ends    = np.where(np.diff(photo_trigger) == -1)[0]

    trial_start    = trial_starts[i_trial + skip_count] - baseline_duration_samples
    trial_duration = int(stim_duration * sampling_rate) + baseline_duration_samples

    # Wait for enough data after trial_start to fill a complete trial
    while eeg.shape[1] < trial_start + trial_duration:
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get()
            eeg       = np.concatenate((eeg, eeg_in),             axis=1)
            aux       = np.concatenate((aux, aux_in),             axis=1)
            timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
        core.wait(0.05)

    print('total: ', eeg.shape, aux.shape, timestamp.shape)
    # mne filter needs enough samples; skip filtering if buffer is too short
    if eeg.shape[1] >= sampling_rate:
        filtered_eeg = mne.filter.filter_data(eeg, sfreq=sampling_rate, l_freq=2, h_freq=40, verbose=False)
    else:
        filtered_eeg = eeg.copy()
    trial_eeg      = np.copy(filtered_eeg[:, trial_start:trial_start + trial_duration])
    trial_aux      = np.copy(aux[:,          trial_start:trial_start + trial_duration])


    # while len(trial_ends) <= i_trial + skip_count:
    #     while not queue_in.empty():
    #         eeg_in, aux_in, timestamp_in = queue_in.get()
    #         print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
    #         eeg       = np.concatenate((eeg, eeg_in),             axis=1)
    #         aux       = np.concatenate((aux, aux_in),             axis=1)
    #         timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
    #     photo_trigger = (aux[1] > 20).astype(int)
    #     trial_starts  = np.where(np.diff(photo_trigger) == 1)[0]
    #     trial_ends    = np.where(np.diff(photo_trigger) == -1)[0]

    # print('total: ', eeg.shape, aux.shape, timestamp.shape)
    # trial_start    = trial_starts[i_trial + skip_count] - baseline_duration_samples
    # trial_duration = int(stim_duration * sampling_rate) + baseline_duration_samples
    # # mne filter needs enough samples; skip filtering if buffer is too short
    # if eeg.shape[1] >= sampling_rate:
    #     filtered_eeg = mne.filter.filter_data(eeg, sfreq=sampling_rate, l_freq=2, h_freq=40, verbose=False)
    # else:
    #     filtered_eeg = eeg.copy()

    # #filtered_eeg   = mne.filter.filter_data(eeg, sfreq=sampling_rate, l_freq=2, h_freq=40, verbose=False)
    # trial_eeg      = np.copy(filtered_eeg[:, trial_start:trial_start + trial_duration])
    # trial_aux      = np.copy(aux[:,          trial_start:trial_start + trial_duration])
    print(f'trial {i_trial}: ', trial_eeg.shape, trial_aux.shape)

    # Force exact trial length so all trials have identical shape
    if trial_eeg.shape[1] < trial_duration:
        pad_n = trial_duration - trial_eeg.shape[1]
        trial_eeg = np.pad(trial_eeg, ((0,0),(0,pad_n)), mode='constant')
        trial_aux = np.pad(trial_aux, ((0,0),(0,pad_n)), mode='constant')
    elif trial_eeg.shape[1] > trial_duration:
        trial_eeg = trial_eeg[:, :trial_duration]
        trial_aux = trial_aux[:, :trial_duration]
    
    print(f'trial {i_trial}: ', trial_eeg.shape, trial_aux.shape)
    baseline_average = np.mean(trial_eeg[:, :baseline_duration_samples], axis=1, keepdims=True)

#    print(f'trial {i_trial}: ', trial_eeg.shape, trial_aux.shape)
#    baseline_average = np.mean(trial_eeg[:, :baseline_duration_samples], axis=1, keepdims=True)
    trial_eeg       -= baseline_average
    eeg_trials.append(trial_eeg)
    aux_trials.append(trial_aux)
    return trial_eeg

# game phase helper methods

def game_over_screen():
    msg = visual.TextStim(
        window,
        text="GAME OVER\n\nPress SPACE to restart\nPress ESC to quit",
        height=0.09, color="white", pos=(0, 0), alignText="center",
    )
    while True:
        keys = event.getKeys()
        if "escape" in keys:
            quit_clean()
        if "space" in keys:
            return
        window.color = [-1, -1, -1]
        msg.draw()
        window.flip()

def obstacle_warning_phase():
    blocked = set(random.sample(range(LANES), num_obstacles))
    spawn_obstacles(blocked)
    t0 = core.getTime()
    while core.getTime() - t0 < 0.8:
        if "escape" in event.getKeys(keyList=["escape"]):
            quit_clean()
        for i, r in enumerate(lane_rects):
            r.fillColor = [1, 1, -1] if i in blocked else [-1, -1, -1]
        draw_scene(current_lane, status_text=f"MODE: {MODE} | Warning")
        window.flip()
    return blocked

def pre_stimulus_phase(cue_lane=None, trials_remaining=None):
    status = f"Trials left: {trials_remaining}" if trials_remaining is not None else "Get ready..."
    for _ in range(2):
        for r in lane_rects:
            r.fillColor = [1, 1, 1]
        draw_scene(current_lane, status_text=status, cue_lane=cue_lane)
        window.flip()
        core.wait(0.15)
        for r in lane_rects:
            r.fillColor = [-1, -1, -1]
        draw_scene(current_lane, status_text=status, cue_lane=cue_lane)
        window.flip()
        core.wait(0.15)
    t0 = core.getTime()
    while core.getTime() - t0 < PRE_STIM_PAUSE:
        if "escape" in event.getKeys(keyList=["escape"]):
            quit_clean()
        draw_scene(current_lane, status_text=status, cue_lane=cue_lane)
        window.flip()

def stimulus_phase_testing():
    chosen = current_lane
    t0 = core.getTime()
    while core.getTime() - t0 < stim_duration:
        keys = event.getKeys()
        if "escape" in keys:
            quit_clean()
        for r in lane_rects:
            r.fillColor = [1, -1, -1]
        for k in keys:
            if k in ["1", "2", "3", "4", "5"]:
                chosen = int(k) - 1
        draw_scene(current_lane, status_text=f"Press 1-5 (chosen: {chosen+1})")
        window.flip()
    return chosen

def stimulus_phase_bci(cue_lane=None, trials_remaining=None):
    global eeg, aux, timestamp
    status = f"Trials left: {trials_remaining}" if trials_remaining is not None else "BCI stimulus (flicker)"
    for i_frame in range(num_frames):
        if "escape" in event.getKeys(keyList=["escape"]):
            save_data()
            quit_clean()
        if queue_in is not None:
            while not queue_in.empty():
                eeg_in, aux_in, timestamp_in = queue_in.get()
                eeg       = np.concatenate((eeg, eeg_in),             axis=1)
                aux       = np.concatenate((aux, aux_in),             axis=1)
                timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
        for j, r in enumerate(lane_rects):
            v = stimulus_frames[i_frame, j]
            r.fillColor = [v, v, v]
        draw_scene(current_lane, status_text=status, cue_lane=cue_lane)
        if cyton_in or CALIBRATION_MODE:
            photosensor_dot.fillColor = [1, 1, 1]
            photosensor_dot.draw()
        window.flip()

def movement_phase(new_lane):
    global current_lane, crash_count
    current_lane = int(new_lane)
    for r in lane_rects:
        r.fillColor = [-1, -1, -1]
    t0     = core.getTime()
    last_t = t0
    speed  = 1.6
    while core.getTime() - t0 < 1.2:
        if "escape" in event.getKeys(keyList=["escape"]):
            save_data()
            quit_clean()
        now    = core.getTime()
        dt     = now - last_t
        last_t = now
        update_obstacles(dx=speed * dt)
        if not CALIBRATION_MODE and check_collision(current_lane):
            crash_count += 1
            draw_scene(current_lane, status_text=f"CRASH! total crashes: {crash_count}")
            window.flip()
            core.wait(0.2)
            return True
        draw_scene(current_lane, status_text=f"Lane {current_lane+1} | crashes: {crash_count}")
        window.flip()
    return False

# Main game loop
try:
    trial_index = 0

    if CALIBRATION_MODE:
        # calibration loop, mirrors original's `for i_trial, target_id in enumerate(trial_sequence)`
        for i_trial, cue_lane in enumerate(trial_sequence):
            trials_remaining = len(trial_sequence) - i_trial
            labels.append(int(cue_lane))
            print(f"Trial {i_trial+1}/{len(trial_sequence)} — cue lane: {cue_lane+1}")
            pre_stimulus_phase(cue_lane=cue_lane, trials_remaining=trials_remaining)
            stimulus_phase_bci(cue_lane=cue_lane, trials_remaining=trials_remaining)

            trial_eeg = collect_trial_eeg(i_trial)
            target_lane = int(cue_lane)

            movement_phase(target_lane)

        # Auto-save and completion screen
        save_data()
        print(f"Calibration complete! {len(trial_sequence)} trials saved to {save_dir}")
        done_msg = visual.TextStim(
            window,
            text=f"Calibration complete!\n{len(trial_sequence)} trials saved.\n\nPress ESC to exit.",
            height=0.08, color="white", pos=(0, 0), alignText="center",
        )
        while "escape" not in event.getKeys(keyList=["escape"]):
            window.color = [-1, -1, -1]
            done_msg.draw()
            window.flip()
        quit_clean()

    elif MODE == "testing":
        while True:
            event.clearEvents()
            obstacle_warning_phase()
            pre_stimulus_phase()
            target_lane = stimulus_phase_testing()
            crashed = movement_phase(target_lane)
            if crashed:
                game_over_screen()
                crash_count = 0
                current_lane = 2

    else:  # MODE == "bci", not calibration
        while True:
            event.clearEvents()
            obstacle_warning_phase()
            pre_stimulus_phase()
            stimulus_phase_bci()
            trial_eeg = collect_trial_eeg(trial_index)
            trial_index += 1
            if trial_eeg is not None and model is not None:
                cropped_eeg = trial_eeg[:, baseline_duration_samples:]
                prediction  = model.predict(cropped_eeg)[0]
                target_lane = int(prediction) % LANES
            else:
                target_lane = random.randint(0, LANES - 1)
            crashed = movement_phase(target_lane)
            if crashed:
                save_data()
                game_over_screen()
                crash_count  = 0
                current_lane = 2

except Exception as e:
    print("Crash:", repr(e))
    try:
        save_data()
    except Exception:
        pass
    quit_clean()
