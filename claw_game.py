import cv2
import mediapipe as mp
import numpy as np
import robosuite as suite

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from hand_detection import MODEL_PATH, get_palm_center_and_open_state, draw_landmarks_on_frame
from policies import Policy


def run_claw_game():
    """
    Main loop: use webcam hand tracking to control the robosuite robot.

    - Open hand: move end-effector in x/y at a fixed z (claw-machine joystick).
    - Close hand (fist): lower and close gripper to attempt a grasp.
    """
    # Set up robosuite environment
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        render_camera="frontview",
    )

    obs = env.reset()
    policy = Policy(obs)
    initial_cube_z = obs["cube_pos"][2]


    # Set up MediaPipe hand landmarker (reuses the model + constants from hand_detection.py)
    mp_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    mp_options = vision.HandLandmarkerOptions(
        base_options=mp_base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    game_message = ""

    with vision.HandLandmarker.create_from_options(mp_options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for a "mirror" effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Prepare frame for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks_on_frame(frame, landmarks)

                (u, v), is_open = get_palm_center_and_open_state(landmarks)

                # Visualize palm center + state
                cx, cy = int(u * w), int(v * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                state_text = "OPEN" if is_open else "CLOSED"
                color = (0, 255, 0) if is_open else (0, 0, 255)
                cv2.putText(
                    frame,
                    f"{state_text} ({u:.2f}, {v:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )

                # Update policy target based on current hand state
                policy.update_from_hand(u, v, is_open, obs)
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            # Step the environment using the policy
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()

            # If the pickup sequence is done, decide success / failure and end the game.
            if policy.is_pickup_done():
                final_cube_z = obs["cube_pos"][2]
                lifted_enough = final_cube_z > initial_cube_z + 0.05
                if lifted_enough:
                    game_message = "SUCCESS: You picked up the block!"
                    print(game_message)
                else:
                    game_message = "FAIL: Block not lifted. Try again next time."
                    print(game_message)
                break

            # Show the webcam feed for debugging + feedback
            if game_message:
                cv2.putText(
                    frame,
                    game_message,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Hand Tracking - Claw Game", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Allow manual quit at any time
            if done:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_claw_game()

