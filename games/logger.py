#%%
import json

class StepRecord:
    def __init__(self, state, action, reward, next_state, terminated, truncated, info):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def __repr__(self):
        return (f"StepRecord(observation={self.state}, action={self.action}, "
                f"reward={self.reward}, next_observation={self.next_state}, "
                f"terminated={self.terminated}, truncated={self.truncated}, info={self.info})")
    
    def to_dict(self):
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info
        }

    
class EpisodeRecorder:
    def __init__(self):
        self.records = []

    def record_state(self, state, action, reward, next_state, terminated, truncated, info):
        step_record = StepRecord(state, action, reward, next_state, terminated, truncated, info)
        self.records.append(step_record)

    def get_records(self):
        return self.records
    
    def save_episode(self, file_path):
        with open(file_path, mode='w') as file:
            json.dump([record.to_dict() for record in self.records], file, indent=4)

    def reset(self):
        self.records = []
    