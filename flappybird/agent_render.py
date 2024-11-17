import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from models import DQN
from models import NeuralNetwork
from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium
import os
import pygame

DATE_FORMAT = "%m-%d %H:%M:%S"
run_folder = datetime.now().strftime('%y%m%d_%H%M')
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

        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            #policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            #policy_dqn.eval()
            pass

        pygame.init()
        clock = pygame.time.Clock()

        for episode in itertools.count():
            if episode % 10000 == 0:
                print(f"Episode {episode} started")
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while(not terminated and episode_reward < self.stop_on_reward):
                if render:
                    env.render()
                    clock.tick(60)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        env.close()
                        return

                keys = pygame.key.get_pressed()
                action = 1 if keys[pygame.K_SPACE] else 0
                action = torch.tensor(action, dtype=torch.int64, device=device)

                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)
            print(f"episode Reward {episode_reward}")

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        env.close()
        pygame.quit()

    def save_graph(self, rewards_per_episode, epsilon_history):
        pass
        # Method implementation here

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        pass
        # Method implementation here

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
