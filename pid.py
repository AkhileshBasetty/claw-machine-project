import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target):
        # Initialize gains and setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.reset(target)

    def reset(self, target=None):
        # Reset internal error history
        self.error = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.target = target
        self.integral = np.zeros(3)
        
    def get_error(self):
       # Return magnitude of last error
        return np.linalg.norm(self.error)

    def update(self, current_pos, dt):
        # Compute and return control signal
        """
            [Short Description]: returns pid controller output value and updates the error state.
            [AI Declaration]: Generated using IntelliCode with the prompt: "complete the pid function"
            Args:
            parameter_1 (type): current position.
            parameter_2 (type): time interval.
            Returns:
            float: control signal value.
            Notes:
            Edited it to use class variables for error terms.
        """
        self.error = self.target - current_pos
        self.integral += self.error * dt
        error_derivative = (self.error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = self.error
        return self.kp * self.error + self.ki * self.integral + self.kd * error_derivative