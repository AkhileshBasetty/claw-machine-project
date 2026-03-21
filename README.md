# CS 188 Final Project - Vision-Controlled Claw Machine

### Group Members

- Akhilesh Basetty
- Jiali Chen
- Omar Lejmi

## Project Overview

This project recreates an arcade-style claw machine game in simulation using a Franka Panda robot in Robosuite's `Lift` task.

Instead of controlling the robot with a physical joystick, the main interaction mode uses webcam-based hand tracking:

- **Open hand**: move the end-effector over the table.
- **Close fist**: trigger a grab sequence (descend, close gripper, and lift).

The game is considered a **win** if the cube is lifted above its initial height threshold after the pickup sequence.

## Key Features

- Real-time hand tracking with Google MediaPipe.
- Gesture-based open/closed hand state detection.
- PID-based Cartesian control of robot end-effector position.
- Randomized cube spawn location each game.
- Side-by-side gameplay with optional keyboard baseline control.
- Evaluation script to compare keyboard vs hand-tracking performance across multiple rounds.

## How It Works

The main gameplay loop is in `claw_game.py`:

1. Create a Robosuite `Lift` environment with randomized cube placement.
2. Track hand landmarks from webcam frames (or keyboard input in baseline mode).
3. Map normalized input coordinates to table-space XY targets.
4. Use `Policy` (`policies.py`) to produce robot actions:
   - **Joystick mode**: track XY at fixed hover Z with open gripper.
   - **Pickup sequence**: descend, close gripper, then lift.
5. Render simulation to an OpenCV fullscreen window.
6. End with success/failure once pickup sequence completes.

## Repository Structure

- `claw_game.py` - Main game loop and environment setup.
- `policies.py` - State-machine policy and PID target logic.
- `pid.py` - PID controller implementation.
- `hand_detection.py` - MediaPipe utilities and hand model setup.
- `evaluate_claw_modes.py` - Evaluation harness for keyboard vs hand modes.
- `evaluation_results.json` - Structured evaluation logs.
- `evaluation_results.txt` - Human-readable evaluation logs.

## Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install numpy opencv-python mediapipe robosuite
```

Depending on your system, Robosuite may also require MuJoCo/OpenGL-related system dependencies. If Robosuite import or rendering fails, verify your MuJoCo and graphics setup for Linux.

### 3) Webcam access

Hand-tracking mode requires a working webcam (`cv2.VideoCapture(0)`).

## Running the Project

### Run the claw game (hand-tracking mode)

```bash
python claw_game.py
```

### Run the evaluation script (keyboard vs hand)

```bash
python evaluate_claw_modes.py
```

The evaluation script runs:

- 5 keyboard-controlled games
- 5 hand-tracking-controlled games

It appends results to `evaluation_results.txt` and `evaluation_results.json`.

## Controls

### Hand mode

- Move hand in camera view to control XY position.
- Keep hand **open** to continue moving.
- Make a **fist** (open -> closed transition) to trigger grab.
- Press `q` to quit current game.

### Keyboard mode (used by evaluation script)

- `W/A/S/D` or arrow keys: move target position.
- `Space`: trigger grab.
- `q`: quit current game.

## Evaluation Metrics

Each session tracks:

- wins
- total games
- accuracy percentage
- quits
- hand-minus-keyboard accuracy difference

In this project, quits are counted as losses for accuracy calculations.

## Notes and Assumptions

- Cube placement is randomized at reset.
- The control policy uses a fixed hover height (`fixed_z`) in joystick mode.
- Success is determined by whether cube height increased enough after pickup.
- The hand landmark model file (`hand_landmarker.task`) is downloaded automatically on first run.

## Troubleshooting

- **Webcam fails to open**: ensure no other app is using the camera and test with a simple OpenCV capture script.
- **No hand detected reliably**: improve lighting, keep one hand in frame, and avoid motion blur.
- **Robot misses cube often**: tune gesture timing and hand alignment; small XY offsets can cause misses.
- **Dependency errors**: verify virtual environment is active and reinstall required packages.
- **Robosuite rendering/import issues**: check MuJoCo and Linux graphics dependencies.

## Demos

#### Winning the game:
[Screencast from 03-03-2026 04:58:39 PM.webm](https://github.com/user-attachments/assets/1d9e7f95-b016-4c0a-9312-7fd905a45c8d)

#### Losing the game:
[Screencast from 03-03-2026 05:00:18 PM.webm](https://github.com/user-attachments/assets/44794c27-fa2d-4e51-93de-311999f66a7f)

