"""Central settings for the sign language recognition project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "landmarks.csv"
MODEL_FILE = MODEL_DIR / "sign_model.joblib"
CONFUSION_MATRIX_FILE = MODEL_DIR / "confusion_matrix.png"

# Signs to recognise. J and Z are left out because they need hand motion.
# TIP: for a quick class demo, shorten this list (e.g. ["A", "B", "C", "L", "V", "Y"]).
LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]

# Camera
CAMERA_INDEX = 0  # try 1 if the wrong / no camera opens
FULLSCREEN = True  # True = video fills the whole screen, False = normal small window

# MediaPipe hand tracking
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Data collection
SAMPLES_PER_CLASS = 200   # samples to record for each sign
CAPTURE_INTERVAL = 0.05   # seconds between saved samples while capturing

# 21 hand landmarks x (x, y, z)
NUM_FEATURES = 63

# Live recognition
PREDICTION_CONFIDENCE_THRESHOLD = 0.60  # ignore predictions below this
SMOOTHING_WINDOW = 10                   # frames used for majority voting
HOLD_SECONDS = 1.5                      # hold a sign this long to type the letter
