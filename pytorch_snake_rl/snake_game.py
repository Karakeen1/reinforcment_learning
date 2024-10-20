# https://github.com/patrickloeber/python-fun/tree/master/snake-pygame
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 15)
#font = pygame.font.SysFont('arial', 25)


# adjustment to the snake_game.py for reinforcment learning 
# reset
# reward

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
# SPEED = 5 # slow
# SPEED = 20 # human control
SPEED = 1000 # for faster training

class SnakeGameAI:
    
    def __init__(self, w=640/2, h=480/2):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
      
    def reset(self):  
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),   # comment / uncomment for more/less starting body parts
                      #Point(self.head.x-(2*BLOCK_SIZE), self.head.y),
                      #Point(self.head.x-(2*BLOCK_SIZE), self.head.y),
                      #Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action, n_games, high_score, reward):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0.1 # going further means more points
        game_over = False
        if len(self.snake) < 4:
            bodyparts = 4
        else:
            bodyparts = len(self.snake)
            
        if self.is_collision() or self.frame_iteration > 100*bodyparts: # after some iteration of "doing nothing" the game stops
            if self.frame_iteration > 100*len(self.snake):
                print("Game over: out of time")
                reward = -100
            elif self.head.x >= self.w or self.head.x < 0 or self.head.y >= self.h or self.head.y < 0:
                print("Game over: snake hit the boundary")
                reward = -10
            else:
                print("Game over: snake bit itself")
                reward = -10
            game_over = True
            
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 1000
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui(n_games, high_score)
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self, n_games, high_score):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [3, 3])
        text_high_score = font.render("High Score: " + str(high_score), True, WHITE)
        self.display.blit(text_high_score, [3, 23])
        text_n_game = font.render("Game: " + str(n_games), True, WHITE)
        self.display.blit(text_n_game, [3, 43])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change of dirction, this is straight
        if np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn right -> down -> left -> up
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn  r -> u -> l -> d   
            
        self.direction = new_dir     
            
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)