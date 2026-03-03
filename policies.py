import numpy as np
from pid import PID

class Policy(object):
    
    def __init__(self, obs):
        # Initial target: centered in the middle of the table at a fixed z-height,
        # regardless of where the cube is.
        cube_pos = obs["cube_pos"]
        eef_pos = obs["robot0_eef_pos"].copy()

        # Keep a fixed z for "joystick" control; start at current eef z (not tied to cube)
        self.fixed_z = eef_pos[2]

        # Start in the middle of the table (global origin in Lift env)
        center_xy = np.array([0.0, 0.0])
        self.target = np.array([center_xy[0], center_xy[1], self.fixed_z])
        self.pid = PID(kp=5.0, ki=0.0, kd=0.0, target=self.target)

        # Gripper: negative -> open, positive -> close (robosuite convention)
        self.gripper_command = 1.0

        # Track last hand state to detect "closing" gesture edges
        self.last_is_open = True

        # Simple state machine:
        # "joystick"       -> hand controls x/y at fixed z
        # "pickup_descend" -> open gripper, move down to cube
        # "pickup_close"   -> close gripper around cube
        # "pickup_lift"    -> lift cube up
        # "pickup_done"    -> hold pose; game logic checks success and ends
        self.mode = "joystick"
        self.pickup_counter = 0

        # XY position at which the pickup sequence starts (where the user aimed)
        self.pickup_xy = self.target[:2].copy()
        # Z hover position when pickup starts; we lift back to this
        self.pickup_hover_z = self.target[2]

        # Store cube height at initialization to compare later
        self.initial_cube_z = cube_pos[2]
        
    def get_action(self, obs):
        """
        Compute the low-level action using PID toward the current 3D target.
        The first 3 entries are Cartesian deltas, last entry is gripper command.
        """
        current_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]

        # Update state machine targets for pickup sequence
        if self.mode == "pickup_descend":
            # Open gripper and move straight down (keep the XY where the user stopped)
            self.gripper_command = -1.0
            lower_z = cube_pos[2] + 0.015
            self.target = np.array([self.pickup_xy[0], self.pickup_xy[1], lower_z])
            self.pid.target = self.target
            if np.linalg.norm(current_pos - self.target) < 0.01:
                self.mode = "pickup_close"
                self.pickup_counter = 0

        elif self.mode == "pickup_close":
            # Close gripper for a short duration while staying at the contact point
            self.gripper_command = 1.0
            lower_z = cube_pos[2] + 0.015
            self.target = np.array([self.pickup_xy[0], self.pickup_xy[1], lower_z])
            self.pid.target = self.target
            self.pickup_counter += 1
            if self.pickup_counter > 10:
                self.mode = "pickup_lift"
                self.pickup_counter = 0

        elif self.mode == "pickup_lift":
            # Lift the cube up while keeping the gripper closed
            self.gripper_command = 1.0
            # Lift back to the hover height we had when pickup started
            lift_z = self.pickup_hover_z
            self.target = np.array([self.pickup_xy[0], self.pickup_xy[1], lift_z])
            self.pid.target = self.target
            if np.linalg.norm(current_pos - self.target) < 0.02:
                self.mode = "pickup_done"

        control_signal = self.pid.update(current_pos, 0.01)
        return np.array(
            [
                control_signal[0],
                control_signal[1],
                control_signal[2],
                0,
                0,
                0,
                self.gripper_command,
            ]
        )

    def update_from_hand(self, u, v, is_open, obs, xy_range=0.15):
        """
        Update the PID target and gripper command based on hand position / state.

        - (u, v) are normalized image coordinates in [0, 1].
        - When the hand is open, the end-effector tracks x/y at a fixed z.
        - When the hand transitions from open -> closed (fist), we:
          * start lowering in z toward the cube
          * close the gripper.
        """
        cube_pos = obs["cube_pos"]

        # Once we've entered the pickup sequence, ignore further hand input.
        if self.mode != "joystick":
            return

        # Map screen coordinates to a square region around the cube in the table plane.
        # Swap axes so that moving your hand left/right moves the arm left/right
        # in the rendered view (instead of up/down).
        x_center, y_center = cube_pos[0], cube_pos[1]
        # Invert vertical mapping so moving hand up moves arm "back"
        x = x_center + (v - 0.5) * 2.0 * xy_range
        y = y_center + (u - 0.5) * 2.0 * xy_range

        if is_open:
            # Joystick mode: move in x/y, keep a fixed z and open gripper.
            self.fixed_z = max(self.fixed_z, cube_pos[2] + 0.15)
            self.target = np.array([x, y, self.fixed_z])
            self.pid.target = self.target
            self.gripper_command = -1.0
        else:
            # Detect a rising edge of "fist" (open -> closed) to start the grab.
            if self.last_is_open:
                # Lock in the current joystick XY and hover Z as the pickup pose
                self.pickup_xy = self.target[:2].copy()
                self.pickup_hover_z = self.target[2]
                self.mode = "pickup_descend"

        self.last_is_open = is_open

    def is_pickup_done(self):
        return self.mode == "pickup_done"

