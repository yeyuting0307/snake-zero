#%%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import numpy as np

sys.path.append(os.path.join(os.pardir, 'games'))
episode_path = os.path.join(os.pardir, "logs", "episode_1718900227.npz")
episodes = np.load(episode_path, allow_pickle=True).get("episode")
episodes
#%%
from tqdm import tqdm
from mlx import data as dx
episodes
#%%
valid_vector = [
    { 
        "state" : episode.state,
        "action1" : episode.action1,
        "reward1"  : episode.reward1,
        "done" : int(episode.terminated or episode.truncated),
        "next_state" : episode.next_state,
    } for episode in tqdm(episodes)
]

#%%
tr = dx.buffer_from_vector(valid_vector)
tr

#%%