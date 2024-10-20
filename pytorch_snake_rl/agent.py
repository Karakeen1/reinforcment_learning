import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import time

MAX_MEMORY = 100_000
BATCH_SIZE = 2048
LEARNING_RATE = 0.005


class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # parameter for randomness
        self.gamma = 0.96 # discount rate # play arround between 0.8 to 0.99, Tutorial was 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # if memory exceeded it popleft()
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down
        ]

        return np.array(state, dtype=int)

    
    def remember(self, state, action, reward, next_state, done): # done = game over
        self.memory.append((state, action, reward, next_state, done)) # extra brackets ( .. ) for storing a tuple, popleft() if Max_MEMORY is recieved
    
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples 
        else:
            mini_sample = self.memory
         
        states, actions, rewards, next_states, dones = zip(*mini_sample)    
        self.trainer.train_step(states, actions, rewards, next_states, dones)
            
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    
    def get_action(self, state, random_count, predicted_count):
        final_move = [0, 0, 0] # intitialice moving array
        # in beginning: random moves: tradeoff exploration / exploitation
        self.epsilon = int(1000/(1+self.n_games)+1)
        if random.randint(0,100) < self.epsilon: # the more games, the lesser it enters this if state, so random becomes absent
            move = random.randint(0, 4)
            if move > 2:
                move = 0 # going mor often straight then taking a turn, should prefenting eating herself
            final_move[move] = 1
            random_count += 1
            last_move_was = "random"            
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            predicted_count += 1     
            last_move_was = "predicted"  
        return final_move, random_count, predicted_count, last_move_was
    
            
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0 
    record = 0 
    agent = Agent()
    game = SnakeGameAI()
    random_count = 0
    predicted_count = 0
    reward = 0
    
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        #get move
        final_move, random_count, predicted_count, last_move_was = agent.get_action(state_old, random_count, predicted_count)
        
        #perform the move
        reward, done, score = game.play_step(final_move, agent.n_games, record, reward)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory, also called replay memory or experienced replay memory
            # plot results
            game.reset()
            
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                # Convert the record to a string and check its length
                record_str = str(record)
                if len(record_str) < 2:
                    record_str = f"0{record_str}"
                
                model_file_name = f"model_hs{record_str}_g{agent.n_games}.pth"
                agent.model.save(model_file_name)
                
            print("Game:", agent.n_games, "Score:", score, "Record:", record, "Reward:", reward)
            print("Random/Predicted Moves", random_count,":", predicted_count, "Last move was:",last_move_was,"\n*****************")
            random_count = 0
            predicted_count = 0
            reward = 0
            
            # plot results
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)                           


if __name__ == "__main__":
    train()