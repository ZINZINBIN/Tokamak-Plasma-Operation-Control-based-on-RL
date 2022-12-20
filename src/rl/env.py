''' Custom Environment for tokamak operation control
    This enviornment is based on Neural Network which predict the next state of the tokamak plasma from being trained by KSTAR dataset
    Current version : prediction of the 0D parameters such as beta, q95, li, ne
    Later version : prediction of the 0D paramters + Magnetic surface (Grad-Shafranov solver)
    Reference
    - https://www.gymlibrary.dev/content/environment_creation/
    - https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
    - https://medium.com/cloudcraftz/build-a-custom-environment-using-openai-gym-for-reinforcement-learning-56d7a5aa827b
    - https://github.com/notadamking/Stock-Trading-Environment
    - https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
'''
import gym
import os, subprocess, time, signal
import torch
import torch.nn as nn
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from src.rl.rewards import RewardSender
import logging

logger = logging.getLogger(__name__)

class NeuralEnv(gym.Env):
    metadata = {'render.modes':['human']} # what does it means?
    def __init__(self, predictor : nn.Module, device : str, reward_sender : RewardSender):
        super().__init__()
        # predictor : output as next state of plasma
        self.predictor = predictor.to(device)
        self.predictor.eval()
        self.device = device
        
        # reward engineering
        self.reward_sender = reward_sender
        
        # initalize state, action, done
        self.done = False
        self.init_state = None
        self.init_action = None
        
        self.state = None
        self.action = None
    
    # update initial condition for plasma operation
    def update_init_state(self, init_state : torch.Tensor, init_action : torch.Tensor):
        self.init_state = init_state.to(self.device)
        self.init_action = init_action.to(self.device)
        
        self.state = init_state.to(self.device)
    
    def reset(self):
        self.done = False
        self.state = self.init_state
        self.action = self.init_action
        
        return self.state

    def step(self, action : torch.Tensor):
        state = self.state
        
        if len(state.size()) == 2:
            state = state.unsqueeze(0)
        
        if len(action.size()) == 2:
            action = action.unsqueeze(0)
            
        inputs = torch.concat([state, action], axis = 2)
        next_state = self.predictor(inputs)
        reward = self.reward_sender(next_state)
        
        self.check_terminal_state(next_state)
        
        self.state = next_state
        
        return next_state, reward, self.done, {}

    def get_reward(self, next_state : torch.Tensor):
        return self.reward_sender(next_state)
    
    def check_terminal_state(self, next_state : torch.Tensor):
        # if state contains nan value, then terminate the environment
        if torch.isnan(next_state).sum() > 0:
            self.done = True