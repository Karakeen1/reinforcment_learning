import gym
import numpy as np
import time

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000

SHOW_EVERY = 200

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

render = False

for episode in range(EPISODES):
    print(f"Episode: {episode}")
    if episode % SHOW_EVERY == 0 and episode != 0:
        print(f"Render @ Episode {episode}")
        env = gym.make("MountainCar-v0", render_mode="human")
        env.reset()
    else:
        env = gym.make("MountainCar-v0")
        env.reset()

    discrete_state = get_discrete_state(env.reset()[0])
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()


        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print(f"We reached the goal on episode {episode}")

        discrete_state = new_discrete_state

env.close()
