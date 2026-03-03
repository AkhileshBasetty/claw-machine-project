import numpy as np
from pid import PID

class Policy(object):
    
    def __init__(self, obs):
        print(obs)
        self.pid = PID(kp=1.0, ki=0.0, kd=0.0, target=obs['cube_pos'])
        self.gripper_command = 0.0
        
    def get_action(self, obs):
        current_pos = obs['robot0_eef_pos']
        control_signal = self.pid.update(current_pos, 0.01)
        return np.array([control_signal[0], control_signal[1], control_signal[2], 0, 0, 0, self.gripper_command])
