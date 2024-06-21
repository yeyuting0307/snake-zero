#%%
import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from mlx import data as dx
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.pardir, 'games'))

ACTION_MAP = {
    None : 0, 
    'up' : 1,
    'down' : 2, 
    'left' : 3, 
    'right' : 4, 
}

def episode_loader(episode_paths, action_map=ACTION_MAP):
    episodes = []
    for episode_path in episode_paths:
        episodes.extend(np.load(episode_path, allow_pickle=True).get("episode"))
    valid_vector = [
        { 
            "state" : episode.state,
            "action1" : action_map[episode.action1],
            "reward1"  : episode.reward1,
            "done" : int(episode.terminated or episode.truncated),
            "next_state" : episode.next_state,
        } for episode in tqdm(episodes)
    ]

    tr = dx.buffer_from_vector(valid_vector)

    tr_iter = (
        tr.shuffle()\
        .to_stream()\
        .key_transform("state", lambda x: x.astype("float32") / 255.0)\
        .batch(32)\
        .prefetch(4, 4)
    )
    return tr_iter

#%%



# %%
