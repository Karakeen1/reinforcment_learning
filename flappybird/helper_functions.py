# Define memory for Experience Replay
from collections import deque
import random
import torch

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
    
    
class FrameStack42:
    def __init__(self, num_stack=4, device='cpu'):
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        self.device = device

    def reset(self):
        for _ in range(self.num_stack):
            self.frames.append(torch.zeros(1, 42, 42, device=self.device))
        return self.get_state()

    def add_frame(self, frame):
        assert frame.shape == (1, 42, 42), "Frame should be of shape (1, 42, 42)"
        self.frames.append(frame.to(self.device))
        return self.get_state()

    def get_state(self):
        return torch.cat(list(self.frames), dim=0)
    
    
class FrameStack:
    def __init__(self, num_stack=4, device='cpu'):
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        self.device = device 

    def reset(self):
        for _ in range(self.num_stack):
            self.frames.append(torch.zeros(1, 84, 84, device=self.device))
        return self.get_state()

    def add_frame(self, frame):
        assert frame.shape == (1, 84, 84), "Frame should be of shape (1, 84, 84)"
        self.frames.append(frame.to(self.device))
        return self.get_state()

    def get_state(self):
        return torch.cat(list(self.frames), dim=0)