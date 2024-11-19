# Define memory for Experience Replay
from collections import deque
import random
class ReplayMemory_original():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
    
    
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def push(self, state, action, new_state, reward, terminated):
        self.memory.append((state, action, new_state, reward, terminated))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)