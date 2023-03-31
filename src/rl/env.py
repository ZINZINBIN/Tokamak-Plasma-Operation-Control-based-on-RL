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
import pandas as pd
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from src.rl.rewards import RewardSender
from typing import Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)

class NeuralEnv(gym.Env):
    metadata = {'render.modes':['human']} # what does it means?
    def __init__(
        self, 
        predictor : nn.Module, 
        device : str, 
        reward_sender : RewardSender, 
        seq_len : int, 
        pred_len : int, 
        range_info : Dict, 
        t_terminal : float = 4.0, 
        dt : float = 0.01, 
        cols_control = None,
        limit_ctrl_rate : bool = False,
        rate_range_info : Optional[Dict] = None,
        ):
        
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
        
        # current state 
        self.t_released = 0 
        self.current_state = None
        self.current_action = None
        
        # information for virtual operation
        self.dt = dt # time interval
        self.t_terminal = t_terminal # terminal time
        
        # input sequence length
        self.seq_len = seq_len
        
        # output sequence length
        self.pred_len = pred_len    
        
        # range information about action space
        self.action_space = {
            'low' : [range_info[col][0] for col in range_info.keys()],
            'upper' : [range_info[col][1] for col in range_info.keys()],
        }
        
        # original shot info : used for playing policy network
        self.original_shot = None
        
        # columns info
        self.cols_control = cols_control
        
        # control rate limit
        self.limit_ctrl_rate = limit_ctrl_rate
        
        if limit_ctrl_rate:
            self.action_space['rate-low'] = [rate_range_info[col][0] for col in rate_range_info.keys()]
            self.action_space['rate-upper'] = [rate_range_info[col][1] for col in rate_range_info.keys()]
        
    def load_shot_info(self, df : pd.DataFrame):
        self.original_shot = df
    
    # update initial condition for plasma operation
    def update_init_state(self, init_state : torch.Tensor, init_action : torch.Tensor):
        
        if init_state.ndim == 2:
            init_state = init_state.unsqueeze(0)
            
        if init_action.ndim == 2:
            init_action = init_action.unsqueeze(0)
        
        self.init_state = init_state
        self.init_action = init_action
        
        self.current_state = init_state
        self.current_action = init_action
        
    def update_state(self, next_state : torch.Tensor):
        state = self.current_state
        
        if next_state.ndim == 2:
            next_state = next_state.unsqueeze(0)
        
        next_state = torch.concat([state, next_state], axis = 1)
        self.current_state = next_state[:,-self.seq_len:,:]
        
        # time sync
        self.t_released += self.dt * self.pred_len
        
    def update_action(self, next_action : torch.Tensor):
        action = self.current_action
        
        if next_action.ndim == 2:
            next_action = next_action.unsqueeze(0)
        
        next_action = torch.concat([action, next_action], axis = 1)
        self.current_action = next_action[:,-self.seq_len-self.pred_len:,:]
        
    def get_state(self):
        return self.current_state

    def get_action(self):
        return self.current_action
    
    def reset(self):
        self.done = False
        self.current_state = self.init_state
        self.current_action = self.init_action
        self.t_released = 0
        
        return self.current_state

    def step(self, action : torch.Tensor):
        # get state
        state = self.get_state()
        
        if state.ndim == 2:
            state = state.unsqueeze(0)
        
        if action.ndim == 2:
            action = action.unsqueeze(0)
            
        # update action
        self.update_action(action)
        action = self.get_action()
            
        # next state        
        next_state = self.predictor(state.to(self.device), action.to(self.device)).detach().cpu()
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
        self.current_state.cpu()
        self.init_state.cpu()
        
        self.current_action.cpu()
        self.init_action.cpu()
        
        self.predictor.cpu()
        
        self.current_state = None
        self.init_state = None
        
        self.current_action = None
        self.init_action = None
        
        # cpu cache memory clear
        gc.collect()
        
        # cuda gpu cache memory clear
        torch.cuda.empty_cache()
        
# This is the custom environment with considering the stochastic behavior of the tokamak operation
class StochasticNeuralEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(
        self, 
        predictor : nn.Module, 
        device : str, 
        reward_sender : RewardSender, 
        seq_len : int, 
        pred_len : int, 
        range_info : Dict, 
        t_terminal : float = 4.0, 
        dt : float = 0.01, 
        cols_control = None,
        limit_ctrl_rate : bool = False,
        rate_range_info : Optional[Dict] = None,
        noise_mean_0D : float = 0,
        noise_std_0D : float = 1.0,
        noise_mean_ctrl : float = 0,
        noise_std_ctrl : float = 1.0,
        scale_0D : float = 1.0,
        scale_ctrl : float = 1.0,
        ):
        
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
        
        # current state 
        self.t_released = 0 
        self.current_state = None
        self.current_action = None
        
        # information for virtual operation
        self.dt = dt # time interval
        self.t_terminal = t_terminal # terminal time
        
        # input sequence length
        self.seq_len = seq_len
        
        # output sequence length
        self.pred_len = pred_len    
        
        # range information about action space
        self.action_space = {
            'low' : [range_info[col][0] for col in range_info.keys()],
            'upper' : [range_info[col][1] for col in range_info.keys()],
        }
        
        # original shot info : used for playing policy network
        self.original_shot = None
        
        # columns info
        self.cols_control = cols_control
        
        # control rate limit
        self.limit_ctrl_rate = limit_ctrl_rate
        
        if limit_ctrl_rate:
            self.action_space['rate-low'] = [rate_range_info[col][0] for col in rate_range_info.keys()]
            self.action_space['rate-upper'] = [rate_range_info[col][1] for col in rate_range_info.keys()]
            
        # Stochastic properties
        self.scale_0D = scale_0D
        self.scale_ctrl = scale_ctrl
        
        self.noise_mean_0D = noise_mean_0D
        self.noise_mean_ctrl = noise_mean_ctrl
        
        self.noise_std_0D = noise_std_0D
        self.noise_std_ctrl = noise_std_ctrl
        
    def add_noise(self, x : torch.Tensor, choice : Literal['state', 'action']):
        
        if choice == 'state':
            noise = torch.ones_like(x).to(x.device) * self.noise_mean_0D + torch.randn(x.size()).to(x.device) * self.noise_std_0D
            noise *= self.scale_0D
            x += noise
        
        elif choice == 'action':
            noise = torch.ones_like(x).to(x.device) * self.noise_mean_ctrl + torch.randn(x.size()).to(x.device) * self.noise_std_ctrl
            noise *= self.scale_ctrl
            x += noise
            
        return x

    def load_shot_info(self, df : pd.DataFrame):
        self.original_shot = df
    
    # update initial condition for plasma operation
    def update_init_state(self, init_state : torch.Tensor, init_action : torch.Tensor):
        
        if init_state.ndim == 2:
            init_state = init_state.unsqueeze(0)
            
        if init_action.ndim == 2:
            init_action = init_action.unsqueeze(0)
        
        self.init_state = init_state
        self.init_action = init_action
        
        self.current_state = init_state
        self.current_action = init_action
        
    def update_state(self, next_state : torch.Tensor):
        state = self.current_state
        
        if next_state.ndim == 2:
            next_state = next_state.unsqueeze(0)
        
        next_state = torch.concat([state, next_state], axis = 1)
        self.current_state = next_state[:,-self.seq_len:,:]
        
        # time sync
        self.t_released += self.dt * self.pred_len
        
    def update_action(self, next_action : torch.Tensor):
        action = self.current_action
        
        if next_action.ndim == 2:
            next_action = next_action.unsqueeze(0)
        
        next_action = torch.concat([action, next_action], axis = 1)
        self.current_action = next_action[:,-self.seq_len-self.pred_len:,:]
        
    def get_state(self):
        return self.current_state

    def get_action(self):
        return self.current_action
    
    def reset(self):
        self.done = False
        self.current_state = self.init_state
        self.current_action = self.init_action
        self.t_released = 0
        
        return self.current_state

    def step(self, action : torch.Tensor):
        # get state
        state = self.get_state()
        
        if state.ndim == 2:
            state = state.unsqueeze(0)
        
        if action.ndim == 2:
            action = action.unsqueeze(0)
            
        # update action
        self.update_action(action)
        action = self.get_action()
        
        # add noise
        state = self.add_noise(state, 'state')
        action = self.add_noise(action, 'action')
            
        # next state        
        next_state = self.predictor(state.to(self.device), action.to(self.device)).detach().cpu()
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
        self.current_state.cpu()
        self.init_state.cpu()
        
        self.current_action.cpu()
        self.init_action.cpu()
        
        self.predictor.cpu()
        
        self.current_state = None
        self.init_state = None
        
        self.current_action = None
        self.init_action = None
        
        # cpu cache memory clear
        gc.collect()
        
        # cuda gpu cache memory clear
        torch.cuda.empty_cache()