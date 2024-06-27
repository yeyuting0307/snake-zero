#%%
from mlx import core as mx
import mlx.nn as nn

# %%
class CNNPolicy(nn.Module):
    def __init__(self, in_channel, output_channel):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
        )
        self.linear = nn.Linear(38400, output_channel)
       
    def __call__(self, x):
        output = self.cnn(x)
        output = mx.flatten(output).reshape(x.shape[0], -1)
        output = self.linear(output)
        return output
    