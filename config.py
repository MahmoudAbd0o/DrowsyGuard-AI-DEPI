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
EAR_CLOSED_THRESH = 0.21      # fallback threshold when calibration is skipped
PERCLOS_WINDOW_SEC = 2.0      # observation window (was 60 frames @ ~30fps)
PERCLOS_HIGH = 40.0           # % closed => drowsy
PERCLOS_LOW = 15.0            # % closed => low risk
PERCLOS_MIN_OBS = 15          # min valid samples before reporting
PERCLOS_MIN_COVERAGE = 0.30   # min valid/total ratio for a trustworthy value

# --------------------------------------------------------- eye calibration --
# Every driver has a different eye shape, so one fixed threshold misfires on
# some faces. The app measures the driver's own open-eye EAR at start-up and
# derives a personal "closed" threshold from it.
EAR_CALIBRATION_SEC = 3.0
EAR_CALIBRATION_MIN_SAMPLES = 10
EAR_CALIBRATION_MIN_BASELINE = 0.18   # lower samples are blinks, not "open"
EAR_CLOSED_RATIO = 0.60               # personal threshold = baseline * ratio
EAR_CLOSED_MIN = 0.13                 # clamp: never miss a real closure
EAR_CLOSED_MAX = 0.26                 # clamp: never flag a normal stare
EAR_SMOOTH_WINDOW = 5                 # rolling median over the last N samples

# ------------------------------------------- when is an EAR trustworthy? --
# A turned head or a tiny/far face deforms the eye corners and fakes closure.
EAR_MAX_TURN_RATIO = 0.22     # yaw above this => measurement not usable
FACE_TOO_FAR_WIDTH_PX = 110   # face narrower than this => "move closer"
DETECTION_CONFIDENCE = 0.35   # MediaPipe: lower => finds smaller/farther faces
TRACKING_CONFIDENCE = 0.40
ZOOM_FACTOR = 2.0             # centre-crop retry when no face is found

# -------------------------------------------------- decision stabilisation --
DROWSY_CONFIRM_SEC = 1.2      # DROWSY must hold this long before the red alarm
DROWSY_HOLD_SEC = 0.8         # keep the red state this long after it clears
DROWSY_EXIT_PERCLOS = 25.0    # ...and only once PERCLOS drops below this

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
FRAME_WIDTH = 1280            # capture/processing resolution (was 640x480)
FRAME_HEIGHT = 720
DISPLAY_WIDTH = 800           # on-screen size (keeps the window small)
DISPLAY_HEIGHT = 450
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
