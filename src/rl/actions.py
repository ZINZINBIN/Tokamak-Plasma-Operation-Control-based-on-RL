import gym
import numpy as np
from typing import List, Dict

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return action
    
# Each control parameter have limit variation over time
# We reduce its incline and decline rates + Maximum and minimum value + On / Off value
class CtrlValScaler:
    def __init__(self, ctrl_setup : Dict):
        self.setup = ctrl_setup
        self.ctrl_names = ctrl_setup.keys()