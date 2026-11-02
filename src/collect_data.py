"""Step 1 - record training samples from your webcam.

Controls (click the video window first):
    SPACE : start / stop capturing for the current sign
    N     : next sign
    P     : previous sign
    R     : redo - delete saved samples of the current sign and record it again
    X     : delete ALL saved samples of ALL signs (asks Y to confirm)
    Q     : quit (data is saved automatically)

Run:  python src/collect_data.py
      python src/collect_data.py --labels A B C --samples 300
"""
import argparse
import csv
import time
from collections import Counter

import cv2

import config
from utils import (create_hands, draw_hand, extract_features, find_hand, open_camera,
                put_text, show_frame)

def load_counts():
    counts = Counter()
    if config.DATA_FILE.exists():
        with open(config.DATA_FILE, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    counts[row[0]] += 1
    return counts


def remove_label(label):
    """Delete all saved samples of one sign from the data file."""
    with open(config.DATA_FILE, newline="") as f:
        rows = list(csv.reader(f))
    kept = rows[:1] + [row for row in rows[1:] if row and row[0] != label]
    with open(config.DATA_FILE, "w", newline="") as f:
        csv.writer(f).writerows(kept)


def clear_all():
    """Delete every saved sample (keeps only the header row)."""
    with open(config.DATA_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["label"] + [f"f{i}" for i in range(config.NUM_FEATURES)])


def main():
    parser = argparse.ArgumentParser(description="Collect sign language landmark data")
    parser.add_argument("--labels", nargs="+", default=config.LABELS, help="signs to record")
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_CLASS,
                        help="samples to record per sign")
    args = parser.parse_args()

    labels = [label.upper() for label in args.labels]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    counts = load_counts()
    new_file = not config.DATA_FILE.exists()
    csv_file = open(config.DATA_FILE, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(config.NUM_FEATURES)])

    cap = open_camera()
    hands = create_hands()
    index = 0
    capturing = False
    confirm_clear = False
    last_save = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from camera.")
                break
            frame = cv2.flip(frame, 1)  # mirror view
            label = labels[index]

            hand = find_hand(hands, frame)
            if hand is not None:
                draw_hand(frame, hand)
                due = time.time() - last_save >= config.CAPTURE_INTERVAL
                if capturing and counts[label] < args.samples and due:
                    writer.writerow([label] + extract_features(hand).tolist())
                    counts[label] += 1
                    last_save = time.time()

            done = counts[label] >= args.samples
            if done:
                capturing = False

            put_text(frame, f"Sign: {label}  ({index + 1}/{len(labels)})", (10, 35), 1.0, (255, 255, 0))
            put_text(frame, f"Samples: {counts[label]}/{args.samples}", (10, 70))
            if done:
                status, color = "DONE - press N for next sign", (0, 255, 255)
            elif hand is None:
                status, color = "No hand detected", (0, 0, 255)
            elif capturing:
                status, color = "CAPTURING... move hand slightly", (0, 255, 0)
            else:
                status, color = "Hand OK - press SPACE to capture", (255, 255, 255)
            put_text(frame, status, (10, 105), 0.7, color)
            if confirm_clear:
                put_text(frame, "Delete ALL samples? Y = yes, any other key = cancel",
                         (10, 140), 0.7, (0, 0, 255))
            put_text(frame, "SPACE capture | N next | P prev | R redo | X clear all | Q quit",
                     (10, frame.shape[0] - 15), 0.5, (255, 255, 255), 1)

            show_frame("Collect data", frame)
            key = cv2.waitKey(1) & 0xFF
            if confirm_clear and key != 255:
                if key == ord("y"):
                    csv_file.close()
                    clear_all()
                    counts.clear()
                    csv_file = open(config.DATA_FILE, "a", newline="")
                    writer = csv.writer(csv_file)
                    print("Deleted ALL saved samples.")
                confirm_clear = False
            elif key == ord("q"):
                break
            elif key == ord("x"):
                confirm_clear = True
                capturing = False
            elif key == ord(" "):
                capturing = not capturing
            elif key == ord("n"):
                index = (index + 1) % len(labels)
                capturing = False
            elif key == ord("p"):
                index = (index - 1) % len(labels)
                capturing = False
            elif key == ord("r"):
                csv_file.close()
                remove_label(label)
                counts[label] = 0
                csv_file = open(config.DATA_FILE, "a", newline="")
                writer = csv.writer(csv_file)
                capturing = False
                print(f"Deleted saved samples of {label}. Record it again.")
    finally:
        csv_file.close()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    print("\nSamples saved per sign:")
    for label in labels:
        print(f"  {label}: {counts[label]}")
    print(f"\nData file: {config.DATA_FILE}")


if __name__ == "__main__":
    main()
