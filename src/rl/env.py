''' Custom Environment for tokamak operation control
    This enviornment is based on Neural Network which predict the next state of the tokamak plasma from being trained by KSTAR dataset
    Current version : prediction of the 0D parameters such as beta, q95, li, ne
    Later version : prediction of the 0D paramters + Magnetic surface (Grad-Shafranov solver)
    Reference
    - https://www.gymlibrary.dev/content/environment_creation/
    - https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
'''

import gym
import os, subprocess, time, signal
import torch.nn as nn
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import logging

logger = logging.getLogger(__name__)

class NeuralEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes':['human']} # what does it means?
    def __init__(self, predictor : nn.Module, device : str):
        super().__init__()
        self.predictor = predictor.to(device)
        self.predictor.eval()
        self.device = device
        pass    
    
    def reset(self):
        self.done = False
        self.state = self.start
        return self.state

    def step(self, action):
        assert self.action_space.contains(action)
        pass
    def render(self):
        pass
    def close(self):
        pass
        