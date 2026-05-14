import os
import cv2
import time
import math
from collections import deque
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------- helpers ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def count_extended_fingers(lm):
    wrist = lm[0]
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]  # tip, pip
    extended = 0
    for tip, pip in fingers:
        if dist(lm[tip], wrist) > dist(lm[pip], wrist):
            extended += 1
    return extended

def classify_open_fist(extended_count):
    if extended_count >= 3:
        return "OPEN"
    if extended_count <= 1:
        return "FIST"
    return "UNKNOWN"

def stable_label(history, label, min_count):
    return sum(1 for x in history if x == label) >= min_count


def main():
    # Download this model once (see note below)
    countdown_seconds = 3
    countdown_end_time = 0.0
    countdown_value = 0

    model_path = "hand_landmarker.task"
    output_dir = os.path.join(os.path.dirname(__file__), "photos")
    os.makedirs(output_dir, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # <- key change
    if not cap.isOpened():
    # fallback attempts
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (tried index 0 and 1 with DirectShow).")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    history = deque(maxlen=10)
    open_needed = 8
    fist_needed = 6
    cooldown_seconds = 1.5
    cooldown_until = 0.0
    state = "IDLE"
    last_trigger_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy()

        h, w = frame.shape[:2]

        # Convert to MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Timestamp (ms) required for video mode; we can simulate with time
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        label = "NO_HAND"
        extended = None

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]  # list of 21 landmarks
            extended = count_extended_fingers(lm)
            label = classify_open_fist(extended)

            # Draw landmarks (simple circles)
            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        history.append(label)
        now = time.time()
        in_cooldown = now < cooldown_until

        now = time.time()
        in_cooldown = now < cooldown_until

# -------------------------
# State machine (trigger)
# -------------------------
        if not in_cooldown:
            if state == "IDLE":
                    if stable_label(history, "OPEN", open_needed):
                        state = "OPEN_CONFIRMED"

            elif state == "OPEN_CONFIRMED":
                if stable_label(history, "FIST", fist_needed):
            # Start countdown
                    countdown_end_time = now + countdown_seconds
                    countdown_value = countdown_seconds
                    state = "COUNTDOWN"

# -------------------------
# Countdown handling
# -------------------------
        if state == "COUNTDOWN":
            remaining = int(math.ceil(countdown_end_time - now))
            countdown_value = max(0, remaining)

            if now >= countdown_end_time:
        # Take photo at the end of countdown
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(output_dir, f"gesture_photo_{ts}.jpg")
                cv2.imwrite(filename, clean_frame)

                last_trigger_time = now
                cooldown_until = now + cooldown_seconds
                state = "COOLDOWN"

# -------------------------
# Cooldown exit
# -------------------------
        if state == "COOLDOWN" and not in_cooldown:
            state = "IDLE"

        if state == "COOLDOWN" and not in_cooldown:
            state = "IDLE"

        hud = f"Label: {label}"
        if extended is not None:
            hud += f" | Extended: {extended}/4"
        hud += f" | State: {state}"
        if in_cooldown:
            hud += f" | Cooldown: {max(0, cooldown_until - now):.1f}s"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 255, 30), 2)

        if now - last_trigger_time < 0.25:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 12)
            cv2.putText(frame, "CAPTURED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        if state == "COUNTDOWN":
            cv2.putText(frame, f"{countdown_value}", (w//2 - 20, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8)

        cv2.imshow("Open->Fist Photo Capture (q to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
