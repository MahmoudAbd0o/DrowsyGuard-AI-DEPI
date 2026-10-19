"""
DrowsyGuard AI - central configuration.

Single source of truth for every threshold and constant.
Modules must import from here instead of hard-coding values.
Verified environment: Python 3.10, mediapipe==0.10.9.
"""

# ---------------------------------------------------------------- serial --
SERIAL_PORT = "COM5"          # Arduino port (Device Manager -> COMx)
SERIAL_BAUD = 9600

# ------------------------------------------------------------- eye / EAR --
EAR_CLOSED_THRESH = 0.21      # EAR below this => eye considered closed
PERCLOS_WINDOW_SEC = 2.0      # observation window (was 60 frames @ ~30fps)
PERCLOS_HIGH = 40.0           # % closed => drowsy
PERCLOS_LOW = 15.0            # % closed => low risk
PERCLOS_MIN_OBS = 15          # min valid samples before reporting
PERCLOS_MIN_COVERAGE = 0.30   # min valid/total ratio for a trustworthy value

# ----------------------------------------------------------- mouth / MAR --
MAR_YAWN_THRESH = 0.60        # mouth open ratio => yawning
YAWN_MIN_FRAMES = 12          # frames mouth must stay open => real yawn
YAWN_ALERT_COUNT = 2          # yawns within the window => drowsy contributor
YAWN_WINDOW_SEC = 90.0

# ------------------------------------------------------- head / distract --
HEAD_DOWN_THRESH = 0.55       # nose below eye line, normalised by chin
TURN_ENTER = 0.22             # head-turn ratio => enter distracted
TURN_EXIT = 0.12              # back to centre => start clearing
TURN_MIN_FRAMES = 45          # frames above TURN_ENTER => distracted
TURN_CLEAR_FRAMES = 15        # frames below TURN_EXIT => clear
DISTRACT_COOLDOWN_SEC = 3.0

# ------------------------------------------------------------- attention --
ATT_HIGH = 70                 # attention % >= this => High
ATT_MED = 40                  # attention % >= this => Medium, else Low
ATT_PENALTY_HEAD_DOWN = 10
ATT_PENALTY_DISTRACTED = 15
ATT_PENALTY_YAWN = 8
ATT_PENALTY_YAWN_NOW = 5
ATT_CAP_DROWSY = 35
ATT_CAP_DISTRACTED = 55
ATT_CAP_LOW = 69

# ---------------------------------------------------------------- debounce --
LEVEL_DEBOUNCE_FRAMES = 5     # level must hold this many frames

# ------------------------------------------------------------------ graph --
GRAPH_LEN = 120

# -------------------------------------------------------------- face mesh --
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
MIN_LANDMARKS = 468

# ------------------------------------------------------- driver selection --
MIRROR = True                 # selfie view: screen acts like a mirror
DRIVER_ANCHOR = 0.70          # mirrored view: driver on the right (Egypt)
SIDE_BIAS = 0.5               # driver side vs face size trade-off
PINCH_ANCHOR = 0.30           # non-mirrored anchor (phase1 behaviour)

# ------------------------------------------------------------------ video --
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECORD_FPS = 20.0
AUTOSAVE_SEC = 60.0

# ----------------------------------------------------------------- colours --
BGR_GREEN = (0, 200, 0)
BGR_ORANGE = (0, 165, 255)
BGR_YELLOW = (0, 255, 255)
BGR_RED = (0, 0, 255)
BGR_GRAY = (160, 160, 160)
BGR_WHITE = (255, 255, 255)

ATT_COLORS = {
    "green": BGR_GREEN,
    "orange": BGR_ORANGE,
    "yellow": BGR_YELLOW,
    "red": BGR_RED,
    "gray": BGR_GRAY,
}

# ------------------------------------------------------------------ output --
SESSION_REPORT = "session_report.csv"
EVAL_REPORT = "evaluation_report.csv"
LOG_DIR = "outputs/application_logs"
