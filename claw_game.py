import cv2
import mediapipe as mp
import numpy as np
import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from hand_detection import MODEL_PATH, get_palm_center_and_open_state, draw_landmarks_on_frame
from policies import Policy

SIM_WIDTH = 1920
SIM_HEIGHT = 1080

# Keyboard control
KEYBOARD_STEP = 0.1
KEYBOARD_UV_MIN, KEYBOARD_UV_MAX = 0.15, 0.85

WIN_NAME = "Claw Game"


def _make_env():
    """Create and return a fresh Lift env with random cube placement."""
    cube_sampler = UniformRandomSampler(
        name="CubeSampler",
        mujoco_objects=None,
        x_range=[-0.15, 0.15],
        y_range=[-0.15, 0.15],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0.0, 0.0, 0.8)),
        z_offset=0.01,
    )
    return suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        placement_initializer=cube_sampler,
    )


def run_one_game(control_type="hand"):
    """
    Run a single claw game with either hand tracking or keyboard control.
    Returns True if the cube was lifted (win), False otherwise, None if user quit.
    Uses the same offscreen sim render + fullscreen OpenCV window; no change to rendering logic.
    """
    env = _make_env()
    obs = env.reset()
    policy = Policy(obs)
    initial_cube_z = obs["cube_pos"][2]

    cap = None
    landmarker = None
    key_u, key_v = 0.5, 0.5
    key_trigger_grab = False

    if control_type == "hand":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            env.close()
            raise RuntimeError("Could not open webcam.")
        mp_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        mp_options = vision.HandLandmarkerOptions(
            base_options=mp_base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        landmarker = vision.HandLandmarker.create_from_options(mp_options)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    result = None
    game_message = ""

    try:
        while True:
            u, v, is_open = 0.5, 0.5, True

            if control_type == "hand":
                ret, cam_frame = cap.read()
                if not ret:
                    break
                cam_frame = cv2.flip(cam_frame, 1)
                h, w, _ = cam_frame.shape
                rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                det_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if det_result.hand_landmarks:
                    landmarks = det_result.hand_landmarks[0]
                    draw_landmarks_on_frame(cam_frame, landmarks)
                    (u, v), is_open = get_palm_center_and_open_state(landmarks)
                    cx, cy = int(u * w), int(v * h)
                    cv2.circle(cam_frame, (cx, cy), 10, (0, 255, 0), -1)
                    state_text = "OPEN" if is_open else "CLOSED"
                    color = (0, 255, 0) if is_open else (0, 0, 255)
                    cv2.putText(
                        cam_frame, f"{state_text} ({u:.2f}, {v:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                    )
                    policy.update_from_hand(u, v, is_open, obs)
                else:
                    cv2.putText(
                        cam_frame, "No hand detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    )
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    result = None
                    break
                if key == 83 or key == ord("d"):
                    key_u = min(KEYBOARD_UV_MAX, key_u + KEYBOARD_STEP)
                if key == 81 or key == ord("a"):
                    key_u = max(KEYBOARD_UV_MIN, key_u - KEYBOARD_STEP)
                if key == 82 or key == ord("w"):
                    key_v = max(KEYBOARD_UV_MIN, key_v - KEYBOARD_STEP)
                if key == 84 or key == ord("s"):
                    key_v = min(KEYBOARD_UV_MAX, key_v + KEYBOARD_STEP)
                if key == ord(" "):
                    key_trigger_grab = True
                u, v = key_u, key_v
                is_open = not key_trigger_grab
                if key_trigger_grab:
                    key_trigger_grab = False
                policy.update_from_hand(u, v, is_open, obs)

            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)

            # Render simulation to numpy array (BGR) — same logic as before
            sim_frame = env.sim.render(
                camera_name="frontview",
                height=SIM_HEIGHT,
                width=SIM_WIDTH,
            )[::-1]
            sim_frame = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)

            if control_type == "hand":
                pip_w, pip_h = SIM_WIDTH // 4, SIM_HEIGHT // 4
                cam_small = cv2.resize(cam_frame, (pip_w, pip_h))
                sim_frame[SIM_HEIGHT - pip_h:, SIM_WIDTH - pip_w:] = cam_small

            if game_message:
                cv2.putText(
                    sim_frame, game_message,
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3,
                )

            cv2.imshow(WIN_NAME, sim_frame)

            if policy.is_pickup_done():
                final_cube_z = obs["cube_pos"][2]
                lifted_enough = bool(final_cube_z > initial_cube_z + 0.01)
                result = lifted_enough
                game_message = (
                    "SUCCESS: You picked up the block!"
                    if lifted_enough
                    else "FAIL: Block not lifted. Try again next time."
                )
                print(game_message)
                cv2.putText(
                    sim_frame, game_message,
                    (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3,
                )
                cv2.imshow(WIN_NAME, sim_frame)
                cv2.waitKey(3000)
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                result = None
                break
            if done:
                break
    finally:
        if cap is not None:
            cap.release()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()
        env.close()

    return result


def run_claw_game():
    """Run a single hand-tracking claw game (same as before)."""
    run_one_game(control_type="hand")


if __name__ == "__main__":
    run_claw_game()