"""Quick check that every library is installed and the webcam works.

Run:  python src/check_setup.py
"""
import importlib
import sys

print(f"Python {sys.version.split()[0]}")

packages = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "pyttsx3": "pyttsx3 (optional)",
}

missing = False
for module, name in packages.items():
    try:
        mod = importlib.import_module(module)
        print(f"  OK       {name:<22} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        optional = "optional" in name
        print(f"  {'SKIPPED' if optional else 'MISSING'}  {name:<22} ({exc.__class__.__name__})")
        missing = missing or not optional

if missing:
    raise SystemExit("\nInstall missing packages with:  pip install -r requirements.txt")

import cv2  # noqa: E402

import config  # noqa: E402

cap = cv2.VideoCapture(config.CAMERA_INDEX)
if cap.isOpened() and cap.read()[0]:
    print(f"  OK       webcam (index {config.CAMERA_INDEX})")
else:
    print(f"  PROBLEM  cannot read webcam index {config.CAMERA_INDEX} - try changing CAMERA_INDEX in src/config.py")
cap.release()

print("\nSetup looks good. Next: python src/collect_data.py")
