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
from helper_functions import ReplayMemory, FrameStack
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
device = 'cpu' # uncomment to force cpu
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
        self.epsilon_offset     = hyperparameters['epsilon_offset']         # n episodes with stable espilon = epsilon_init
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

    def run(self, is_training=True, render=False, pretrained_model_path=None):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode='human', **self.env_make_params)
        env.metadata['render_fps'] = 1000
        num_actions = env.action_space.n
        # num_states = env.observation_space.shape[0] # 12
        # num_states = (84,84)

        rewards_per_episode = []
        #policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        policy_cnn = CNN(num_actions).to(device)

        frame_stack = FrameStack(num_stack=4, device=device)
        
        
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
       
        humaninput_episodes = 0  # human control to give a head start
        env_step_count = 0
        
        raise_epsilon = False
        
        
        if humaninput_episodes > 0:
                print(f"starting with human input for {humaninput_episodes} episodes")

        for episode in itertools.count():
            
            print_n_episodes = 1000
            if episode % print_n_episodes == 0:
                if episode == 0:  
                    episode_start_time = datetime.now()
                else:
                    episode_end_time = datetime.now()
                    n_episodes_timer = episode_end_time - episode_start_time
                    n_episodes_timer = int(n_episodes_timer.total_seconds())
                    env_steps_rate = env_step_count / n_episodes_timer
                    print(f"Episode {episode} started, time for {print_n_episodes} episodes: {n_episodes_timer} seconds, Steps: {env_step_count}, Steprate: {env_steps_rate} steps/second")
                    episode_start_time = datetime.now()
                    env_step_count = 0 
            
            # Initialize state_stack with zeros        
            state = frame_stack.reset()  
            new_state = frame_stack.reset()
              
            #state, _ = env.reset()
            env.reset()
            # state = torch.tensor(state, dtype=torch.float, device=device)
            # print(f"state from env: {state}, shape: {state.shape}")
            terminated = False
            episode_reward = 0.0
            
            env.render()
            
            screenshot = pygame.surfarray.array3d(screen)
            screenshot = screenshot.transpose([1, 0, 2])  # Transpose to get the correct orientation
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            
            #cv2.imshow('Screenshot', screenshot)
            frame_array = preprocess_frame(screenshot) # shape tupple (84, 84)          
            
            state = torch.tensor(frame_array, dtype=torch.float, device=device)
            state = preprocess_state(state)  # Now state has shape [1, 1, x, y]
            #print(f"state from image: {state}, shape: {state.shape}")
            state = frame_stack.add_frame(state)
            #print(state.dim())
            # print(f"state from image: {state}, shape: {state.shape}")
            
            
            
            
            if episode % 10 == 0 and humaninput_episodes > episode and episode != 0:
                print(f"{episode} episodes of {humaninput_episodes} humaninput episodes passed")

            while(not terminated and episode_reward < self.stop_on_reward):
            
                ###############################################
                # start RL training after some initial manual rounds
                ###############################################
                if episode < humaninput_episodes:
                    env.metadata['render_fps'] = 30
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
                
                    if episode +1 == humaninput_episodes: # change speed in last human interaction episode
                        print(f"RL takes over")
                        env.metadata['render_fps'] = 1000
                        humaninput_episodes = 0                   
                
                else:
                    # Select action based on epsilon-greedy
                    use_higher_epsilon_after_n_episodes = False
                    if (use_higher_epsilon_after_n_episodes and is_training and epsilon < 0.05 and episode % 10000 <= 1000 and episode % 10000 > 0) or raise_epsilon:
                        if not raise_epsilon:
                            epsilon_checkin = epsilon
                        raise_epsilon = True
                        epsilon = 0.125
                        #print(f"episode % 100 = {episode%100}")
                        if random.random() < epsilon:
                            action = env.action_space.sample()
                            action = torch.tensor(action, dtype=torch.int64, device=device)
                        else:
                        # select best action
                            with torch.no_grad():
                                state = state.unsqueeze(0)  # Add batch dimension
                                # action = policy_cnn(state).squeeze().argmax()
                                action = policy_cnn(state).argmax(dim=-1)
                        if episode % 10000 == 1000:
                            epsilon = epsilon_checkin
                            raise_epsilon = False
                            #print(f"reset epsilon to {epsilon} and set raise_epsilon to {raise_epsilon}")
                                
                    elif is_training and random.random() < epsilon:
                        # select random action
                        action = env.action_space.sample()
                        action = torch.tensor(action, dtype=torch.int64, device=device)
                    else:
                        # select best action
                        with torch.no_grad():
                            # Add batch and channel dimensions
                            # with unsqueeze(0).unsqueeze(0) New shape: (1, 1, x, y)
                            state = state.unsqueeze(0)  # Add batch dimension
                            #print(f"State shape before CNN: {state.shape}")
                            # action = policy_cnn(state).squeeze().argmax()
                            action = policy_cnn(state).argmax(dim=-1)


                # use environment only for termination, all other extracted from image
                env_new_state, env_reward, terminated, truncated, info = env.step(action.item())
                env_step_count += 1
                
                screenshot = pygame.surfarray.array3d(screen)
                screenshot = screenshot.transpose([1, 0, 2])  # Transpose to get the correct orientation
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                
                frame_array = preprocess_frame(screenshot) # shape tupple (x, y)
                
                new_state = torch.tensor(frame_array, dtype=torch.float, device=device)
                new_state = preprocess_state(new_state)  # Now state has shape [1, 1, x, y]
                new_state = frame_stack.add_frame(new_state)
                
                # Analyse frame array for reward:
                # at pixel x= 22 the bird is all black for 84x84
                # bird 42x42 at pixel 11, 3 after cropping left
                white_pixel_count = np.sum(frame_array[:, 3])
                #print(white_pixel_count)
                #time.sleep(0.1)
                # >= 76 pixel bird without pipe (<= 4 for 42x42 bw inverted)
                # <= 35 pixel bird between pipes, goes on for ~13..14 frames (>=29 42x42 bw inverted)
                # == 0 pixel out of bounds

                if terminated:
                    reward = -2 # dying
                elif white_pixel_count == 0:
                    reward = -0.5 # touch the top of the screen
                elif white_pixel_count >= 25 :
                    reward = 0.3 # reward for passing the pipe. total ~4
                else:   
                    reward = 0.1
                #print(reward)
                episode_reward += reward
                # new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                

                if is_training:
                    # memory.append((state, action, new_state, reward, terminated))
                    memory.push(state.squeeze(0), action, new_state.squeeze(0), reward, terminated)  # Store without batch dimension
                    step_count += 1

                state = new_state
                

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

                if episode > self.epsilon_offset and not raise_epsilon: # epsilon decay starts after n episodes offset
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    target_cnn.load_state_dict(policy_cnn.state_dict())
                    step_count = 0

        env.close()
        pygame.quit()
        
        
    def save_graph(self, rewards_per_episode, epsilon_history):
    # Save plots
        fig, ax1 = plt.subplots()  # Create a single plot

        # Calculate mean rewards
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        # Plot raw rewards per episode in green
        ax1.plot(rewards_per_episode, color='g', alpha=0.3, label='Rewards per Episode')

        # Plot mean rewards
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Rewards', color='b')
        ax1.plot(mean_rewards, color='b', label='Mean Rewards')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a second y-axis for epsilon decay
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Epsilon Decay', color='r')
        ax2.plot(epsilon_history, color='r', label='Epsilon Decay')
        ax2.tick_params(axis='y', labelcolor='r')

        # Set title and adjust layout
        plt.title('Rewards and Epsilon Decay')
        fig.tight_layout()  # to ensure everything fits without overlap

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Save plot
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)




    # Optimize policy network
    def optimize(self, mini_batch, policy_cnn, target_cnn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        # Ensure all states are in the correct format
        states = torch.stack([preprocess_state(state) for state in states])
        # Remove the extra dimension
        states = states.squeeze(1)  # This will change shape from [batch_size, 1, 1, 84, 84] to [batch_size, 1, 84, 84]

        # Handle potential inconsistencies in actions
        actions = [action.unsqueeze(0) if action.dim() == 0 else action for action in actions]
        actions = torch.cat(actions).to(device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            # target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_cnn(new_states).max(dim=1)[0]
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
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize
    # resized = cv2.resize(black_white, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Separate color channels
    blue = frame[:, :, 0] # most prommising chanel since pipes and bird very dark
    #green = frame[:, :, 1]
    #red = frame[:, :, 2]
    
    #cv2.imwrite("flappybird_blue.png", blue)
    #cv2.imwrite("flappybird_green.png", green)
    #cv2.imwrite("flappybird_red.png", red)
    #cv2.imwrite("flappybird_original.png", frame)
    
    # Crop top, bottom and left (behind the bird unnecessary information)
    height = blue.shape[0]
    cropped = blue[90:height-150, 60:]
    #cv2.imwrite("flappybird_cropped.png", cropped)
    # Resize
    resized = cv2.resize(cropped, (42, 42), interpolation=cv2.INTER_AREA) # try 42x42 pixels
    #cv2.imwrite("flappybird_resized.png", resized)
    
    
    # Apply threshold
    threshold = 170 
    #black_white = np.where(resized > threshold, 0, 255)
    normalized = np.where(resized > threshold, 0, 1) #  allready normalized 0 for 
    #cv2.imwrite("flappybird_blacknwhite.png", black_white)
    
    # Normalize
    #normalized = black_white / 255.0
    #print(normalized)
    return normalized


def preprocess_state_1frame(state): # used if only 1 frame is passed in CNN
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
    
    
def preprocess_state(state): # used if 4 frames are passed
    if state.dim() == 2:
        # Add channel dimension
        return state.unsqueeze(0)
    elif state.dim() == 3:
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
