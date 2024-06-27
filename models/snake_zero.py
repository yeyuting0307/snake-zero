import os
import sys
from mlx import core as mx
import mlx.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from models.cnn_policy import CNNPolicy
from models.dqn import DQN

class SnakeRL(nn.Module):
    def __init__(self, input_channel, n_actions):
        super(SnakeRL, self).__init__()
        self.cnn_policy = CNNPolicy(input_channel, 512)
        self.dqn = DQN(512, n_actions)
    def __call__(self, x):
        out = self.cnn_policy(x)
        out = self.dqn(out)
        return out