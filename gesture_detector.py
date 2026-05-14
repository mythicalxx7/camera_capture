import cv2
import time
import math
from collections import deque
from datetime import datetime

import mediapipe as mp


# -----------------------
# Helper math
# -----------------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def count_extended_fingers(lm):
    """
    Rule-of-thumb finger extension check using distances to the wrist.
    Works well for open palm vs fist.
    We ignore thumb for robustness across orientations.
    """
    wrist = lm[0]

    # For each finger: (tip_id, pip_id)
    fingers = [
        (8, 6),   # index
        (12, 10), # middle
        (16, 14), # ring
        (20, 18), # pinky
    ]

    extended = 0
    for tip_id, pip_id in fingers:
        if dist(lm[tip_id], wrist) > dist(lm[pip_id], wrist):
            extended += 1
    return extended


def classify_open_fist(extended_count):
    """
    Simple thresholds:
    - OPEN: 3 or 4 fingers extended
    - FIST: 0 or 1 fingers extended
    Otherwise: UNKNOWN (in-between / noisy)
    """
    if extended_count >= 3:
        return "OPEN"
    if extended_count <= 1:
        return "FIST"
    return "UNKNOWN"


def stable_label(history, label, min_count):
    """Return True if 'label' appears at least min_count times in history window."""
    return sum(1 for x in history if x == label) >= min_count


# -----------------------
# Main
# -----------------------
def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing the camera index (0,1,2...).")

    # --- Smoothing / state machine settings ---
    window = 10  # how many recent frame labels to keep
    history = deque(maxlen=window)

    # Stability requirements within the history window
    open_needed = 8   # OPEN must appear >= 8/10 frames to confirm
    fist_needed = 6   # FIST must appear >= 6/10 frames to confirm

    # Cooldown to avoid multiple photos
    cooldown_seconds = 1.5
    cooldown_until = 0.0

    state = "IDLE"  # IDLE -> OPEN_CONFIRMED -> COOLDOWN
    last_trigger_time = 0.0

    # Use a slightly higher detection confidence for fewer false triggers
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror for more intuitive UX (optional)
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label = "NO_HAND"
            extended = None

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                lm = hand_lms.landmark

                extended = count_extended_fingers(lm)
                label = classify_open_fist(extended)

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            history.append(label)

            now = time.time()

            # Cooldown handling
            in_cooldown = now < cooldown_until

            # State machine (only when not in cooldown)
            if not in_cooldown:
                if state == "IDLE":
                    if stable_label(history, "OPEN", open_needed):
                        state = "OPEN_CONFIRMED"

                elif state == "OPEN_CONFIRMED":
                    if stable_label(history, "FIST", fist_needed):
                        # TRIGGER: save photo
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"gesture_photo_{ts}.jpg"
                        cv2.imwrite(filename, frame)

                        last_trigger_time = now
                        cooldown_until = now + cooldown_seconds
                        state = "COOLDOWN"

            # Exit cooldown automatically
            if state == "COOLDOWN" and not in_cooldown:
                state = "IDLE"

            # On-screen HUD
            hud = f"Label: {label}"
            if extended is not None:
                hud += f" | Extended: {extended}/4"
            hud += f" | State: {state}"
            if in_cooldown:
                hud += f" | Cooldown: {max(0, cooldown_until - now):.1f}s"

            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 255, 30), 2)

            # Brief flash indicator after capture
            if now - last_trigger_time < 0.25:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 12)
                cv2.putText(frame, "CAPTURED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            cv2.imshow("Open->Fist Photo Capture (q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
