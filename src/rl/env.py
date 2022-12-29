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
import os, subprocess, time, signal, gc
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
    def __init__(self, predictor : nn.Module, device : str, reward_sender : RewardSender, seq_len : int, pred_len : int):
        super().__init__()
        # predictor : output as next state of plasma
        self.predictor = predictor.to(device)
        self.predictor.eval()
        self.device = device
        
        # reward engineering
        self.reward_sender = reward_sender
        
        # initalize state, action, done, time released
        self.done = False
        self.init_state = None
        self.init_action = None
        
        self.state = None
        self.action = None
        
        self.t_released = 0
        self.dt = 4 * 1 / 210
        
        self.t_terminal = 4
        
        # input sequence length
        self.seq_len = seq_len
        # output sequence length
        self.pred_len = pred_len    
    
    # update initial condition for plasma operation
    def update_init_state(self, init_state : torch.Tensor, init_action : torch.Tensor):
        
        if init_state.ndim == 2:
            init_state = init_state.unsqueeze(0)
            
        if init_action.ndim == 2:
            init_action = init_action.unsqueeze(0)
        
        self.init_state = init_state.to(self.device)
        self.init_action = init_action.to(self.device)
        
        self.state = init_state.to(self.device)
        self.action = init_action.to(self.device)
        
    def update_state(self, next_state : torch.Tensor):
        state = self.state
        next_state = next_state.to(self.device)
        
        if len(next_state.size()) == 2:
            next_state = next_state.unsqueeze(0)
        
        next_state = torch.concat([state, next_state], axis = 1)
        self.state = next_state[:,-self.seq_len:,:]
        
        # time sync
        self.t_released += self.dt
        
    def update_action(self, next_action : torch.Tensor):
        action = self.action
        next_action = next_action.to(self.device)
        
        if len(next_action.size()) == 2:
            next_action = next_action.unsqueeze(0)
        
        next_action = torch.concat([action, next_action], axis = 1)
        self.action = next_action[:,-self.seq_len:,:]
        
    def get_state(self):
        return self.state
    
    def reset(self):
        self.done = False
        self.state = self.init_state
        self.action = self.init_action
        self.t_released = 0
        
        return self.state

    def step(self, action : torch.Tensor):
        state = self.state
        
        if len(state.size()) == 2:
            state = state.unsqueeze(0)
        
        if len(action.size()) == 2:
            action = action.unsqueeze(0)
            
        action = action.to(self.device)
        
        # update action
        self.update_action(action)
        action = self.action
            
        # next input
        inputs = torch.concat([state, action], axis = 2)
        next_state = self.predictor(inputs)
        reward = self.reward_sender(next_state)
        
        # update done
        self.check_terminal_state(next_state)
        
        # update state
        self.update_state(next_state)
        
        return next_state, reward, self.done, {}

    def get_reward(self, next_state : torch.Tensor):
        return self.reward_sender(next_state)
    
    def check_terminal_state(self, next_state : torch.Tensor):
        # if state contains nan value, then terminate the environment
        if torch.isnan(next_state).sum() > 0:
            self.done = True
        
        if self.t_released >= self.t_terminal:
            self.done = True
            
    def close(self):
        
        # gpu -> cpu
        self.state.cpu()
        self.init_state.cpu()
        
        self.action.cpu()
        self.init_action.cpu()
        
        self.predictor.cpu()
        
        self.state = None
        self.init_state = None
        
        self.action = None
        self.init_action = None
        
        # cpu cache memory clear
        gc.collect()
        
        # cuda gpu cache memory clear
        torch.cuda.empty_cache()