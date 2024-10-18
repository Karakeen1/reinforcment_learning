import gym
import pygame
from pygame.locals import *

# Initialize pygame and create a window
pygame.init()
win = pygame.display.set_mode((500, 500))

# Initialize the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()

clock = pygame.time.Clock()
done = False

# Main loop
while True:
    # Poll for events
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True

    if done:
        break

    # Capture keyboard inputs
    keys = pygame.key.get_pressed()

    # Default action (let the pole fall)
    action = 0  # Move left
    if keys[K_LEFT]:
        action = 0  # Move left
    elif keys[K_RIGHT]:
        action = 1  # Move right

    # Step the environment with the chosen action
    _, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    if done:
        # Reset the environment if it's done
        state, _ = env.reset()
        done = False

    # Delay to control game speed
    clock.tick(30)

# Close the environment and pygame
env.close()
pygame.quit()
