from mlx import core as mx
import mlx.nn as nn

class DQN(nn.Module):
    def __init__(self, n_observation, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observation, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()
    
    def __call__(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out
    