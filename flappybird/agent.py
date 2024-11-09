# Tutorial https://www.youtube.com/watch?v=RVMpm86equc

import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch
from dqn import DQN

device = "cuda" if torch.cuda.is_available() else 'cpu'

class Agent:
    def run(self, is_training=True, render = False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n 
        
        policy_dqn = DQN(num_states, num_actions).to_device(device)
        
        obs, _ = env.reset()
        while True:
            # Next action:
            # feed the observation to your agent here)
            action = env.action_space.sample()
            # Processing:
            obs, reward, terminated, _, info = env.step(action)
        
            # Checking if the player is still alive
            if terminated:
                print(info)
                break

        env.close()