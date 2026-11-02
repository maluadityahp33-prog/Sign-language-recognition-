"""Step 3 - live sign language recognition from the webcam.

Hold a sign steady for about 1.5 seconds to type that letter.

Controls (click the video window first):
    SPACE : add a space
    B     : backspace
    C     : clear text
    S     : speak the text (needs pyttsx3)
    Q     : quit

Run:  python src/realtime_recognition.py
"""
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np

import config
from utils import (create_hands, draw_hand, extract_features, find_hand, open_camera,
                   put_text, show_frame)
try:
    import pyttsx3

    tts_engine = pyttsx3.init()
except Exception:  # pyttsx3 missing or no speech engine available
    tts_engine = None


def speak(text):
    if tts_engine is None or not text.strip():
        return
    tts_engine.say(text)
    tts_engine.runAndWait()


def main():
    if not config.MODEL_FILE.exists():
        raise SystemExit("No trained model found. Run  python src/train_model.py  first.")

    bundle = joblib.load(config.MODEL_FILE)
    model = bundle["model"]
    print(f"Loaded model: {bundle['name']} (test accuracy {bundle['test_accuracy']:.3f})")

    cap = open_camera()
    hands = create_hands()

    history = deque(maxlen=config.SMOOTHING_WINDOW)
    text = ""
    hold_label = None
    hold_start = 0.0
    already_added = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            hand = find_hand(hands, frame)
            confidence = 0.0
            if hand is not None:
                draw_hand(frame, hand)
                features = extract_features(hand).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                best = int(np.argmax(probs))
                confidence = float(probs[best])
                label = str(model.classes_[best])
                history.append(label if confidence >= config.PREDICTION_CONFIDENCE_THRESHOLD else None)
            else:
                history.clear()

            # Majority vote over the last few frames -> stable prediction
            stable = None
            if len(history) == history.maxlen:
                top, votes = Counter(history).most_common(1)[0]
                if top is not None and votes >= int(0.7 * history.maxlen):
                    stable = top

            # Hold-to-type logic
            now = time.time()
            progress = 0.0
            if stable is not None and stable == hold_label:
                progress = min((now - hold_start) / config.HOLD_SECONDS, 1.0)
                if progress >= 1.0 and not already_added:
                    text += stable
                    already_added = True
            else:
                hold_label = stable
                hold_start = now
                already_added = False

            # --- Overlay ---
            shown = stable if stable else "-"
            put_text(frame, f"Sign: {shown}   conf: {confidence:.2f}", (10, 35), 1.0, (255, 255, 0))
            bar_w = 200
            cv2.rectangle(frame, (10, 50), (10 + bar_w, 66), (255, 255, 255), 1)
            cv2.rectangle(frame, (10, 50), (10 + int(bar_w * progress), 66), (0, 255, 0), -1)
            put_text(frame, f"Text: {text}", (10, frame.shape[0] - 45), 0.9, (0, 255, 255))
            put_text(frame, "SPACE space | B back | C clear | S speak | Q quit",
                     (10, frame.shape[0] - 15), 0.55, (255, 255, 255), 1)

            show_frame("Sign language recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                text += " "
            elif key == ord("b"):
                text = text[:-1]
            elif key == ord("c"):
                text = ""
            elif key == ord("s"):
                speak(text)
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
