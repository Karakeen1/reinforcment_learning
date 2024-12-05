# Convolutional Network
# https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib as plt
import numpy as np


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn=enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calc Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.output(x)

        return Q
    
    
class CNN_1frame(nn.Module):

    def __init__(self, num_actions):
        super(CNN, self).__init__()
        self.number_of_actions = num_actions # flappybird: num_actions = 2
        """
        self.gamma = 0.99
        self.final_epsilon = 0.00001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 100000
        self.minibatch_size = 32
        """

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=6, stride=3) # in Channel 1 for one b&w frame
      
        self.relu1 = nn.ReLU(inplace=True)
        # only 1 conv layer, deleted layer 2 and 3
        self.fc4 = nn.Linear(32*13*13, 1024) # 2592
           
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(1024, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out
  
  
class CNN_4frames(nn.Module): # modified for 4 frames in stack with 42 pixels input

    def __init__(self, num_actions):
        super(CNN, self).__init__()
        self.number_of_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=6, stride=3)
        self.relu1 = nn.ReLU(inplace=True)

        # Calculate the size of the flattened output
        self.fc_input_dim = self._get_conv_output((4, 42, 42))
        self.fc4 = nn.Linear(self.fc_input_dim, 1024)
        
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(1024, self.number_of_actions)

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output = self.conv1(input)
        output = self.relu1(output)
        return int(np.prod(output.size()))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = out.view(out.size(0), -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out
  
    
class CNN(nn.Module): # for 4 frames 84x84 pixels in stack

    def __init__(self, num_actions):
        super(CNN, self).__init__()
        self.number_of_actions = num_actions # 2 

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(1024, self.number_of_actions) # actions = 2
        
        # image size x times y times channels
        # After conv1: 20x20x32
        # After conv2: 9x9x64
        # After conv3: 7x7x64
        # flattened to a 1D tensor of size 3136 (7 * 7 * 64) passed to fully connected layer fc4.

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size(0), -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out


if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)

