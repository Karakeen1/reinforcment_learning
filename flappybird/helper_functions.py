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
    
    
class FrameStack:
    def __init__(self, num_stack=4):
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

    def reset(self):
        for _ in range(self.num_stack):
            self.frames.append(torch.zeros(1, 32, 32))
        return self.get_state()

    def add_frame(self, frame):
        assert frame.shape == (1, 32, 32), "Frame should be of shape (1, 32, 32)"
        self.frames.append(frame)
        return self.get_state()

    def get_state(self):
        return torch.cat(list(self.frames), dim=0)