#%%
import os
import json
import time
import numpy as np
from collections import deque

class StepRecord:
    def __init__(self, state, action1, action2, reward1, reward2, 
                 next_state, terminated, truncated, info
        ):
        self.state = state
        self.action1 = action1
        self.action2 = action2
        self.reward1 = reward1
        self.reward2 = reward2
        self.next_state = next_state
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def __repr__(self):
        return (f"StepRecord(state={self.state}, "
                f"action1={self.action1}, action2={self.action2}, "
                f"reward1={self.reward1}, reward2={self.reward2}, "
                f"next_state={self.next_state}, "
                f"terminated={self.terminated}, truncated={self.truncated}, "
                f"info={self.info})")
    

class EpisodeRecorder:
    def __init__(self, dir_path=None, max_buffer_len=10):
        self.buffer = deque(maxlen=max_buffer_len)
        self.records = []
        dir_path = dir_path or os.path.join(os.path.dirname(__file__), os.pardir, 'logs')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.save_path = os.path.join(dir_path, f'episode_{round(time.time())}.npz')

    def record_buffer(
            self, 
            state, 
            action1, 
            action2, 
            reward1, 
            reward2, 
            next_state, 
            terminated, 
            truncated, 
            info
        ):
        step_record = StepRecord(
            state, 
            action1, 
            action2, 
            reward1, 
            reward2, 
            next_state, 
            terminated, 
            truncated, 
            info
        )
        self.buffer.append(step_record)

    def record_state(self):
        for step in self.buffer:
            self.records.append(step)
        self.buffer.clear()

    def get_records(self):
        return self.records
    
    def save_episode(self):
        np.savez_compressed(
            self.save_path, 
            episode = self.records
        )
       
    def reset(self):
        self.records = []
    
# %%
