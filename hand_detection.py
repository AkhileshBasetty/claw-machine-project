import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Download model if needed
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH,
    )
    print("Downloaded.")

# Landmark indices (same as before)
WRIST         = 0
THUMB_CMC     = 1
INDEX_MCP     = 5
MIDDLE_MCP    = 9
RING_MCP      = 13
PINKY_MCP     = 17
INDEX_TIP     = 8
MIDDLE_TIP    = 12
RING_TIP      = 16
PINKY_TIP     = 20

def get_palm_center_and_open_state(landmarks):
    """
    [Short Description]: Calculates the palm center position and determines whether the hand is open or closed.
    [AI Declaration]: Generated using Claude with the prompt: "get the palm center and detect open and closed state using mediapipe"
    Args:
        landmarks (list): List of hand landmarks from MediaPipe's HandLandmarker, each with x and y attributes normalized to [0, 1].
    Returns:
        tuple: A tuple of ((u, v), is_open) where (u, v) are the normalized palm center
               coordinates and is_open is True if the hand is open, False if closed.
    """
    palm_ids = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    xs = [landmarks[i].x for i in palm_ids]
    ys = [landmarks[i].y for i in palm_ids]
    u, v = float(np.mean(xs)), float(np.mean(ys))

    wrist_pt = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
    fingertip_ids = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    dists = [np.linalg.norm(np.array([landmarks[i].x, landmarks[i].y]) - wrist_pt)
             for i in fingertip_ids]
    is_open = float(np.mean(dists)) > 0.15
    return (u, v), is_open

def draw_landmarks_on_frame(frame, hand_landmarks):
    """
    [Short Description]: Draws all hand landmarks as circles onto a video frame.
    [AI Declaration]: Generated using Claude with the prompt: "draw mediapipe hand landmarks"
    Args:
        frame (np.ndarray): BGR image frame to draw on, modified in place.
        hand_landmarks (list): List of landmarks from MediaPipe's HandLandmarker, each with x and y attributes normalized to [0, 1].
    Returns:
        None
    """
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

def main():
    """
    [Short Description]: Runs a live webcam hand tracking loop, displaying open/closed state and palm position.
    [AI Declaration]: Generated using Claude with the prompt: "write a script that opens a webcam, runs mediapipe hand landmarker, and shows the open and closed state and palm center coordinates"
    Args:
        None
    Returns:
        None
    """
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks_on_frame(frame, landmarks)

                (u, v), is_open = get_palm_center_and_open_state(landmarks)

                cx, cy = int(u * w), int(v * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                state_text = "OPEN" if is_open else "CLOSED"
                color = (0, 255, 0) if is_open else (0, 0, 255)
                cv2.putText(frame, f"{state_text} ({u:.2f}, {v:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                # --- Plug in robot control here ---
                # if is_open: ...
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()