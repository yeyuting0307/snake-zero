import mlx.core as mx

def select_action(policy_net, state):
    ACTION_MAP = {
        None : 0, 
        'up' : 1,
        'down' : 2, 
        'left' : 3, 
        'right' : 4, 
    }
    ACTION_INDEX_MAP = {v:k for k,v in ACTION_MAP.items()}
    mx_state = mx.array(state)[mx.newaxis].astype(mx.float16)
    action_idx = policy_net(mx_state).argmax(1)
    action = ACTION_INDEX_MAP[action_idx.item()]
    return action
    