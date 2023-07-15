import random, os, pickle
from collections import namedtuple, deque
from typing import Optional

# transition
Transition = namedtuple(
    'Transition',
    ('state', 'action','next_state','reward','done')
)

# save trajectory from buffer
class ReplayBuffer(object):
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def sample(self, batch_size : int):
        return random.sample(self.memory, batch_size)

    def get(self):
        return self.memory.pop()
    
    def save_buffer(self, env_name : str, tag : str = "", save_path : Optional[str] = None):
        
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/', exist_ok=True)

        if save_path is None:
            save_path = "checkpoints/buffer_{}_{}".format(env_name, tag)
            
        print("Process : saving buffer to {}".format(save_path))
        
        with open(save_path, "wb") as f:
            pickle.dump(self.memory, f)
        
    def load_buffer(self, save_path : str):
        print("Process : loading buffer from {}".format(save_path))
        
        with open(save_path, 'rb') as f:
            self.memory = pickle.load(f)
            
    def clear(self):
        self.memory.clear()