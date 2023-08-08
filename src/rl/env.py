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
from typing import Dict, Optional, Literal, List
from src.GSsolver.model import PINN, ContourRegressor
from src.config import Config
import logging

logger = logging.getLogger(__name__)

config = Config()

# function for converting dictionary type of input to the tensor.
def Dict2Tensor(col_state_list : List, col_control_list : List, state : Dict, control : Dict):
    
    state_tensor = []
    control_tensor = []
    
    for idx, col in enumerate(col_state_list):
        v = state[col].view(1,-1, 1)
        state_tensor.append(v)
    
    for idx, col in enumerate(col_control_list):
        v = control[col].view(1,-1,1)
        control_tensor.append(v)
    
    state_tensor = torch.concat(state_tensor, dim = 2)
    control_tensor = torch.concat(control_tensor, dim = 2)
    
    return state_tensor, control_tensor
    

# function for converting tensor type of input to the dictionary type.
def Tensor2Dict(col_state_list : List, col_control_list : List, state : torch.Tensor, control : torch.Tensor):
    
    if state.ndim == 2:
        state = state.unsqueeze(0)
    
    if control.ndim == 2:
        control = control.unsqueeze(0)
    
    state_dict = {}
    control_dict = {}
    
    for idx, col in enumerate(col_state_list):
        state_dict[col] = state[:,:,idx]
    
    for idx, col in enumerate(col_control_list):
        control_dict[col] = control[:,:,idx]
    
    return state_dict, control_dict

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
        shape_predictor : Optional[PINN] = None,
        contour_regressor : Optional[ContourRegressor] = None,
        objective : Literal['params-control', 'shape-control', 'multi-objective'] = 'params-control',
        scaler_0D = None,
        scaler_ctrl = None,
        use_stochastic : bool = False,
        noise_mean_0D : float = 0,
        noise_std_0D : float = 1.0,
        noise_mean_ctrl : float = 0,
        noise_std_ctrl : float = 1.0,
        noise_scale_0D : float = 1.0,
        noise_scale_ctrl : float = 1.0,
        gamma : float = 0.995
        ):
        
        super().__init__()
        
        # predictor : output as next state of plasma
        self.predictor = predictor.to(device)
        
        # objective
        self.objective = objective
        
        # scaler
        self.scaler_0D = scaler_0D
        self.scaler_ctrl = scaler_ctrl
        
        if shape_predictor is not None:
            self.shape_predictor = shape_predictor.to(device)
            self.shape_predictor.eval()
            self.flux = None
        else:
            self.shape_predictor = None
            self.flux = None
        
        if contour_regressor is not None:
            self.contour_regressor = contour_regressor.to(device)
            self.contour_regressor.eval()
            self.contour = None
            self.axis = None
        else:
            self.contour_regressor = None
            self.contour = None
            self.axis = None
          
        # shape information
        self.flux_list = []
        self.contour_list = []
        self.axis_list = []
        self.xpts = []
        self.opts = []
        
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
            
        # stochastic
        self.use_stochastic = use_stochastic
        self.noise_mean_0D = noise_mean_0D
        self.noise_mean_ctrl = noise_mean_ctrl
        
        self.noise_std_0D = noise_std_0D
        self.noise_std_ctrl = noise_std_ctrl
        
        self.noise_scale_0D = noise_scale_0D
        self.noise_scale_ctrl = noise_scale_ctrl
        
        # reward
        self.gamma = gamma
        
    # for general policy improvement linear support
    def update_ls_weight(self, weights:List):
        self.reward_sender.update_target_weight(weights)
    
    # for adding stocastic behavior
    def add_noise(self, x : torch.Tensor, choice : Literal['state','action']):
        if choice == 'state':
            noise = torch.ones_like(x).to(x.device) * self.noise_mean_0D + torch.randn(x.size()).to(x.device) * self.noise_std_0D
            noise *= self.noise_scale_0D
            x += noise
        
        elif choice == 'action':
            noise = torch.ones_like(x).to(x.device) * self.noise_mean_ctrl + torch.randn(x.size()).to(x.device) * self.noise_std_ctrl
            noise *= self.noise_scale_ctrl
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
        
        # if stochastic, add noise
        if self.use_stochastic is True:
            state = self.add_noise(state, 'state')
            action = self.add_noise(action, 'action')
            
        # next state        
        next_state = self.predictor(state.to(self.device), action.to(self.device)).detach().cpu()
        reward = self.reward_sender(next_state)
        
        # predict magnetic flux line
        if self.shape_predictor is not None:
            next_action = action[:,-1,:]
            pinn_state, pinn_action = self.convert_PINN_input(next_state, next_action)
            flux = self.shape_predictor(pinn_state.to(self.device), pinn_action.to(self.device))
            gs_loss = self.shape_predictor.compute_GS_loss(flux)
            self.flux = flux
            self.flux_list.append(flux.squeeze(0).detach().cpu().numpy())
            
            (r_axis, z_axis), _ = self.shape_predictor.find_axis(self.flux, eps = 1e-3)
            xpts = self.shape_predictor.find_xpoints(self.flux, eps = 1e-3)
            
            self.opts.append((r_axis, z_axis))
            self.xpts.append(xpts)
        
        if self.contour_regressor is not None:
            next_action = action[:,-1,:]
            pinn_state, pinn_action = self.convert_PINN_input(next_state, next_action)
            flux = self.shape_predictor(pinn_state.to(self.device), pinn_action.to(self.device))
            contour = self.contour_regressor.compute_rzbdys(flux, pinn_state.to(self.device), pinn_action.to(self.device))
            self.contour = contour
            self.contour_list.append(contour)
            
            cen, rad = self.contour_regressor(flux, pinn_state.to(self.device), pinn_action.to(self.device))
            self.axis_list.append(cen.detach().cpu().squeeze(0).numpy())
            self.axis = cen.detach().cpu().squeeze(0).numpy()
        
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
        
    def convert_PINN_input(self, state : torch.Tensor, action : torch.Tensor):
        state_cols = config.input_params[self.objective]['state']
        control_cols = config.input_params[self.objective]['control']
        
        if self.scaler_0D is not None:
            state = state.view(-1, len(state_cols)).cpu().numpy()
            action = action.view(-1, len(control_cols)).cpu().numpy()
            
            state = self.scaler_0D.inverse_transform(state)
            action = self.scaler_ctrl.inverse_transform(action)
            
            state = torch.from_numpy(state).float()
            action = torch.from_numpy(action).float()
        
        state_dict, action_dict = Tensor2Dict(state_cols, control_cols, state, action)
        
        if self.objective =='params-control':
            PINN_state_cols = config.input_params['GS-solver-params-control']['state']
            PINN_control_cols = config.input_params['GS-solver-params-control']['control']
        else:
            PINN_state_cols = config.input_params['GS-solver']['state']
            PINN_control_cols = config.input_params['GS-solver']['control']
        
        PINN_state = []
        PINN_control = []
        
        for col in PINN_state_cols:
            if col in state_dict.keys():
                state = state_dict[col]
            else:
                state = action_dict[col]
                
            PINN_state.append(state)
            
        for col in PINN_control_cols:
            if col in action_dict.keys():
                action = action_dict[col]
            else:
                action = state_dict[col]

            PINN_control.append(action)
            
        ndim = state.ndim
        
        PINN_state = torch.concat(PINN_state, dim = ndim - 1)
        PINN_control = torch.concat(PINN_control, dim = ndim - 1)
        
        if PINN_state.ndim == 3:
            PINN_state = PINN_state.squeeze(1)
        
        if PINN_control.ndim == 3:
            PINN_control = PINN_control.squeeze(1)
            
        # scale adjustment
        PINN_state[:,0] = PINN_state[:,0] * (-1)
        PINN_control = PINN_control
            
        return PINN_state, PINN_control