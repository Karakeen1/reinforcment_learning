import flappy_bird_gymnasium
import gymnasium
import keyboard
import numpy as np

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()
    if True:
        if keyboard.is_pressed('space'):
            action = np.array([1])
        else:
            action = np.array([0])
    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        print(info)
        break

env.close()