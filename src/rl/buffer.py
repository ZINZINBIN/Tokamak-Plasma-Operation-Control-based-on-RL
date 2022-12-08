import numpy as np
import random

# transition
Transition = namedtuple(
    'Transition',
    ('state', 'action','next_state', 'reward', 'done')
)

# save trajectory from buffer
class ReplayBuffer(object):
    def __init__(self, capacity : int):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def  sample(self, batch_size : int):
        return random.sample(self.memory, batch_size)

    def get(self):
        return self.memory.pop()