# CS 188 Final Project - Claw Machine
### Group Members: Akhilesh Basetty, Jiali Chen, Omar Rhemeni

We implemented our variation of an arcade claw machine game. Instead of using a joystick to control a claw, users can lift up their hand to guide a robot arm above a block. When the player thinks they are on top of the block, they close their fist to trigger the robot arm to lower and try grabbing the block. If the block is picked up, the player wins the game. However, if the robot fails to pick up the block, the player loses.

### Implementation

Our Panda robot runs in the robosuite "Lift" environment. We also randomize the location of the cube on the table. To track the hand and detect open/closed hand gestures, we used Google's MediaPipe library. We convert the hand position to a position above the table and move the end effector to that position using PID control. When the hand closes, it initiates a basic policy to lower, grip, and raise the end effector.


### Demos

#### Winning the game:
[Screencast from 03-03-2026 04:58:39 PM.webm](https://github.com/user-attachments/assets/1d9e7f95-b016-4c0a-9312-7fd905a45c8d)

#### Losing the game:
[Screencast from 03-03-2026 05:00:18 PM.webm](https://github.com/user-attachments/assets/44794c27-fa2d-4e51-93de-311999f66a7f)

