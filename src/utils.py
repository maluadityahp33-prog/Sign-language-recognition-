"""Shared helpers: camera, MediaPipe hand detection and feature extraction.

Works with both MediaPipe APIs:
  * older releases (mp.solutions.hands)
  * newer releases (Tasks API / HandLandmarker) - the model file
    models/hand_landmarker.task is downloaded automatically on first run.
"""
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

import config

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_FILE = config.MODEL_DIR / "hand_landmarker.task"


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {config.CAMERA_INDEX}. "
            "Close other apps using the webcam or change CAMERA_INDEX in src/config.py."
        )
    return cap


class _Landmarks:
    """Minimal wrapper so both MediaPipe APIs give us `.landmark` (x, y, z)."""

    def __init__(self, landmarks):
        self.landmark = landmarks


class _SolutionsDetector:
    """Older MediaPipe: mp.solutions.hands."""

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if results.multi_hand_landmarks:
            return _Landmarks(results.multi_hand_landmarks[0].landmark)
        return None

    def close(self):
        self._hands.close()


class _TasksDetector:
    """Newer MediaPipe: Tasks API HandLandmarker."""

    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_FILE)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

    @staticmethod
    def _ensure_model():
        if HAND_MODEL_FILE.exists():
            return
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading MediaPipe hand model (one time, ~8 MB)...")
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_FILE)
        except Exception as exc:
            HAND_MODEL_FILE.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the hand model. Download it manually from\n"
                f"  {HAND_MODEL_URL}\n"
                f"and save it as  {HAND_MODEL_FILE}"
            ) from exc

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must keep increasing
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if result.hand_landmarks:
            return _Landmarks(result.hand_landmarks[0])
        return None

    def close(self):
        self._landmarker.close()


def create_hands():
    """Create a hand detector that tracks one hand (works on any MediaPipe version)."""
    if getattr(mp, "solutions", None) is not None:
        return _SolutionsDetector()
    return _TasksDetector()


def find_hand(hands, frame_bgr):
    """Return the landmarks of the first detected hand, or None."""
    return hands.detect(frame_bgr)


def draw_hand(frame_bgr, hand_landmarks):
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame_bgr, p, 4, (0, 0, 255), -1)


def extract_features(hand_landmarks):
    """Turn 21 landmarks into a 63-value vector that ignores hand position and size.

    1. Move the origin to the wrist (landmark 0) -> position independent.
    2. Divide by the largest wrist-to-landmark distance -> size/distance independent.
    """
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts /= scale
    return pts.flatten()


def put_text(frame, text, org, scale=0.8, color=(0, 255, 0), thickness=2):
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3
    )
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
_screen_size = None
_created_windows = set()


def get_screen_size():
    """Screen resolution in pixels (falls back to 1920x1080 if it cannot be read)."""
    global _screen_size
    if _screen_size is None:
        try:
            import ctypes

            user32 = ctypes.windll.user32  # Windows only
            user32.SetProcessDPIAware()
            _screen_size = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        except Exception:
            _screen_size = (1920, 1080)
    return _screen_size


def show_frame(window_name, frame):
    """Show a frame; fills the whole screen when config.FULLSCREEN is True."""
    if not config.FULLSCREEN:
        cv2.imshow(window_name, frame)
        return

    if window_name not in _created_windows:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        _created_windows.add(window_name)

    screen_w, screen_h = get_screen_size()
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x0, y0 = (screen_w - new_w) // 2, (screen_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    cv2.imshow(window_name, canvas)