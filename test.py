import numpy as np
import robosuite as suite
from policies import *


# create environment instance
env = suite.make(
    env_name="Lift",
    robots="Panda",  
    has_renderer=True,
    render_camera="birdview",
)

# reset the environment
for _ in range(5):
    obs = env.reset()
    policy = Policy(obs)
    
    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment
        
        env.render()  # render on display
        if reward == 1.0 or done: break
