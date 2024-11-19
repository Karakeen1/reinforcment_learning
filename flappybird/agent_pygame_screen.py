# Tutorial: https://www.youtube.com/watch?v=arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi
# https://github.com/markub3327/flappy-bird-gymnasium/blob/main/README.md
# https://github.com/johnnycode8/dqn_pytorch/blob/main/agent.py
# https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial <- convolutional network


import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from models import DQN, CNN
from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium
import os
import pygame
import time
import keyboard

# printscreen
import cv2
from PIL import Image
import pygetwindow as gw
import pyautogui



DATE_FORMAT = "%m-%d %H:%M:%S"
run_folder = "" #
run_folder = datetime.now().strftime('%y%m%d_%H%M') # uncomment to save in a run sub folder with date
RUNS_DIR = f"C:\\Users\\mmose\\OneDrive\\Programmieren\\reinforcment_learning\\flappybird\\runs\\{run_folder}"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(f"using {device}") 

class Agent():
    def __init__(self, hyperparameter_set="flappybird1"):
        with open("C:\\Users\\mmose\\OneDrive\\Programmieren\\reinforcment_learning\\flappybird\\hyperparameters.yaml", 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode='human', **self.env_make_params)
        num_actions = env.action_space.n
        # num_states = env.observation_space.shape[0] # 12
        num_states = (84,84)

        rewards_per_episode = []
        #policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        policy_cnn = CNN(num_actions).to(device)
        
        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            # target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            # target_dqn.load_state_dict(policy_dqn.state_dict())
            target_cnn = CNN(num_actions).to(device)
            target_cnn.load_state_dict(policy_cnn.state_dict())
            self.optimizer = torch.optim.Adam(policy_cnn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            policy_cnn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_cnn.eval()

        # initialize pygame 
        pygame.init()
        pygame.display.set_caption("flappybird")
        screen = pygame.display.set_mode((288, 512))  # Adjust size to match your Flappy Bird window
        clock = pygame.time.Clock()
        saved_image = False
        

        for episode in itertools.count():
            print_n_episodes = 100
            if episode % print_n_episodes == 0:
                if episode == 0:  
                    episode_start_time = datetime.now()
                else:
                    episode_end_time = datetime.now()
                    n_episodes_timer = episode_end_time - episode_start_time
                    print(f"Episode {episode} started, latest episode reward {episode_reward}, time for {print_n_episodes} episodes: {int(n_episodes_timer.total_seconds())} seconds")
                    episode_start_time = datetime.now()
                
            state, _ = env.reset()
            
            # state = torch.tensor(state, dtype=torch.float, device=device)
            # print(f"state from env: {state}, shape: {state.shape}")
            terminated = False
            episode_reward = 0.0
            

            while(not terminated and episode_reward < self.stop_on_reward):
                env.render()
                clock.tick(60)
                
                screenshot = pygame.surfarray.array3d(screen)
                screenshot = screenshot.transpose([1, 0, 2])  # Transpose to get the correct orientation
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                
                #cv2.imshow('Screenshot', screenshot)
                frame_array = preprocess_frame(screenshot) # shape tupple (84, 84)
                # print(frame_array.shape)
                if(not saved_image and episode_reward > 12):
                   cv2.imwrite("flappybird.png", screenshot)
                   saved_image = True
                
                state = torch.tensor(frame_array, dtype=torch.float, device=device)
                state = preprocess_state(state)  # Now state has shape [1, 1, 84, 84]
                # print(f"state from image: {state}, shape: {state.shape}")
                
                # start RL training after some initial manual rounds
                if episode < 100:
                    # Set default action to 0 (do nothing)
                    action = 0

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                action = 1  # Flap

                    # Convert action to tensor
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                
                else:
                    # Select action based on epsilon-greedy
                    if is_training and random.random() < epsilon:
                        # select random action
                        action = env.action_space.sample()
                        action = torch.tensor(action, dtype=torch.int64, device=device)
                    else:
                        # select best action
                        with torch.no_grad():
                            # Add batch and channel dimensions
                            # with unsqueeze(0).unsqueeze(0) New shape: (1, 1, 84, 84)
                            action = policy_cnn(state).squeeze().argmax()

                # use environment for terminated, else is from image
                new_state, reward, terminated, truncated, info = env.step(action.item())
                reward = 0.1
                episode_reward += reward
                # new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # memory.append((state, action, new_state, reward, terminated))
                    memory.push(state.squeeze(0), action, reward, terminated)  # Store without batch dimension
                    step_count += 1

                # state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_cnn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                    
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_cnn, target_cnn)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    target_cnn.load_state_dict(policy_cnn.state_dict())
                    step_count = 0

        env.close()
        pygame.quit()
        
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Optimize policy network
    def optimize(self, mini_batch, policy_cnn, target_cnn):

        # Transpose the list of experiences and separate each element
        #states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states, actions, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        # Ensure all states are in the correct format
        states = torch.stack([preprocess_state(state) for state in states])
        # Remove the extra dimension
        states = states.squeeze(1)  # This will change shape from [batch_size, 1, 1, 84, 84] to [batch_size, 1, 84, 84]

        actions = torch.stack(actions)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            # target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_cnn(states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        # Calcuate Q values from current policy
        current_q = policy_cnn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases



class WindowCapture:
    def get_window(self, window_name = "flappybird"):
        self.window = gw.getWindowsWithTitle(window_name)[0]
        if not self.window:
            raise Exception(f'Window not found: {window_name}')

    def get_screenshot(self):
        # Activate the window
        self.window.activate()

        # Get window position and size
        left, top = self.window.topleft
        right, bottom = self.window.bottomright
        width = right - left
        height = bottom - top

        # Capture screenshot
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
    
        # Convert to numpy array
        return np.array(screenshot)
   
    
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized / 255.0
    
    return normalized


def preprocess_state(state):
    if state.dim() == 2:
        # Add channel and batch dimensions
        return state.unsqueeze(0).unsqueeze(0)
    elif state.dim() == 3:
        # Add batch dimension
        return state.unsqueeze(0)
    elif state.dim() == 4:
        # Already in correct format
        return state
    else:
        raise ValueError(f"Unexpected state shape: {state.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
