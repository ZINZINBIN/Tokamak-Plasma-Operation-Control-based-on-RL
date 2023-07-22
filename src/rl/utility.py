import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Literal, Dict, Union
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from skimage import measure
from src.rl.env import NeuralEnv
from src.GSsolver.util import draw_KSTAR_limiter, modify_resolution
from matplotlib import colors, cm

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def preparing_initial_dataset(
    df : pd.DataFrame,
    cols_0D : List,
    cols_ctrl : List,
    scaler : Literal['Robust', 'Standard', 'MinMax'] = 'Robust'
    ):
    
    # nan handling for cols_ctrl
    df[cols_ctrl] = df[cols_ctrl].fillna(0)

    # nan interpolation for cols_0D
    df.interpolate(method = 'linear', limit_direction = 'forward')

    ts_cols = cols_0D + cols_ctrl
    
    # float type
    for col in ts_cols:
        df[col] = df[col].astype(np.float32)

    if scaler == 'Standard':
        scaler_0D = StandardScaler()
        scaler_ctrl = StandardScaler()
    elif scaler == 'Robust':
        scaler_0D = RobustScaler()
        scaler_ctrl = RobustScaler()
    elif scaler == 'MinMax':
        scaler_0D = MinMaxScaler()
        scaler_ctrl = MinMaxScaler()
  
    df[cols_0D] = scaler_0D.fit_transform(df[cols_0D].values)
    df[cols_ctrl] = scaler_ctrl.fit_transform(df[cols_ctrl].values)
        
    return df, scaler_0D, scaler_ctrl

# get maximum value and minimum value of action space
def get_range_of_output(df : pd.DataFrame, cols_ctrl : List):
    
    range_info = {}

    for col in cols_ctrl:
        min_val = df[col].min()
        max_val = df[col].max()
        range_info[col] = [min_val, max_val]
    
    return range_info

# class for generating the initial state of the plasma and inital control values
# initial state : (1, seq_len, n_0D_parameters)
# initial control value : (1, seq_len + pred_len, n_ctrl_parameters)
class InitGenerator:
    def __init__(self, df : pd.DataFrame, t_init : float, state_cols : List, control_cols : List, seq_len : int, pred_len : int, random : bool = False, shot_num : Optional[int] = None):
        self.df = df
        self.t_init = t_init
        self.state_cols = state_cols
        self.control_cols = control_cols
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # if deterministic
        self.shot_num = shot_num
        self.shot_list = np.unique(df.shot)
        
        # if stochastic
        self.random = random
        
    def get_state(self):
        
        if self.random or self.shot_num is None:
            shot_num = self.shot_list[int(np.random.randint(len(self.shot_list)))]
            df_shot = self.df[self.df.shot == shot_num]
            
            init_indices_0D = df_shot[df_shot.time >= self.t_init].index[0:self.seq_len].values
            init_indices_ctrl = df_shot[df_shot.time >= self.t_init].index[0:self.seq_len+self.pred_len].values
            
            init_state = df_shot[self.state_cols].loc[init_indices_0D].values
            init_action = df_shot[self.control_cols].loc[init_indices_ctrl].values
            
        else:
            shot_num = self.shot_num
            df_shot = self.df[self.df.shot == shot_num]
            
            init_indices_0D = df_shot[df_shot.time >= self.t_init].index[0:self.seq_len].values
            init_indices_ctrl = df_shot[df_shot.time >= self.t_init].index[0:self.seq_len+self.pred_len].values
            
            init_state = df_shot[self.state_cols].loc[init_indices_0D].values
            init_action = df_shot[self.control_cols].loc[init_indices_ctrl].values
            
        init_state = torch.from_numpy(init_state).unsqueeze(0)
        init_action = torch.from_numpy(init_action).unsqueeze(0)
            
        return init_state, init_action
    
# print the result of the RL training
def plot_rl_status(target_value_result : Dict, episode_reward : Dict, tag : str, cols2str : Optional[Dict] = None, save_dir : Optional[str] = None):
    
    fig = plt.figure(figsize = (12,6))
    fig.suptitle("{} Target values and reward per episode".format(tag))
    
    gs = GridSpec(nrows = len(target_value_result.keys()), ncols = 2)

    n_episode = None
    
    for idx, key in enumerate(target_value_result.keys()):
        ax = fig.add_subplot(gs[idx,0])
        target_per_episode = target_value_result[key]
        upper = target_per_episode['max']
        lower = target_per_episode['min']
        mean = target_per_episode['mean']
        
        n_episode = range(1, len(mean) + 1, 1)
        
        clr = plt.cm.Purples(0.9)
        ax.set_facecolor(plt.cm.Blues(0.2))
        ax.plot(n_episode, mean, label = 'target', color = clr)
        ax.fill_between(n_episode, lower, upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
        
        y_name = cols2str[key] if cols2str else key
        x_name = 'episode'
        ax.set_ylabel(y_name)
    
    mean = episode_reward['mean']
    upper = episode_reward['max']
    lower = episode_reward['min']
    
    ax = fig.add_subplot(gs[:,1])
    
    clr = plt.cm.Purples(0.9)
    ax.set_facecolor(plt.cm.Blues(0.2))
    ax.plot(n_episode, mean, label = 'reward', color = clr)
    ax.fill_between(n_episode, lower, upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
    ax.set_xlabel("episode")
    ax.set_ylabel("mean reward")
    
    fig.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir)
        
def plot_virtual_operation(
    env : NeuralEnv,
    state_list : List[np.array], 
    action_list : List[np.array], 
    reward_list : List[np.array],
    seq_len : int, 
    pred_len : int, 
    shot_num : int,
    targets_dict : Dict,
    col2str : Dict,
    scaler_0D : Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]= None,
    scaler_ctrl : Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]= None,
    tag : str = "",
    save_dir : str = "",
    vis_cols : Optional[List] = None
    ):
    
    total_state = None
    total_action = None
    
    for state, action in zip(state_list, action_list):
        if total_state is None:
            total_state = state.reshape(seq_len,-1)
            total_action = action.reshape(seq_len + pred_len,-1)
        else:
            total_state = np.concatenate((total_state, state[-pred_len:,:].reshape(pred_len,-1)), axis= 0)
            total_action = np.concatenate((total_action, action[:,:].reshape(pred_len,-1)), axis= 0)

    # re-scaling : considering the physical unit and scale of the system
    total_state = scaler_0D.inverse_transform(total_state)
    total_action = scaler_ctrl.inverse_transform(total_action)

    # 0D parameter plot
    title = "{}_shot_{}_operation_0D".format(tag, shot_num)
    save_file = os.path.join(save_dir, "{}.png".format(title))
    
    if env.shape_predictor is not None:
        fig = plt.figure(figsize = (16,6), facecolor="white")
    else:
        fig = plt.figure(figsize = (12,6), facecolor="white") 
        
    fig.suptitle(title)
    
    total_cols = env.reward_sender.total_cols
    target_cols = env.reward_sender.targets_cols
    target_cols_indice = env.reward_sender.target_cols_indices
    cols_control = env.cols_control
    
    dt = env.dt
    df_shot = env.original_shot
    
    # rescaling the real experimental value
    df_shot[total_cols] = scaler_0D.inverse_transform(df_shot[total_cols].values)
    df_shot[cols_control] = scaler_ctrl.inverse_transform(df_shot[cols_control].values)
    
    if env.shape_predictor is not None:
        gs = GridSpec(nrows = len(total_cols), ncols = 3)
        
        if vis_cols is not None:
            gs = GridSpec(nrows = len(vis_cols), ncols = 3)
            
    else:
        gs = GridSpec(nrows = len(total_cols), ncols = 2)
        
        if vis_cols is not None:
            gs = GridSpec(nrows = len(vis_cols), ncols = 2)

    for i, col in enumerate(total_cols):
        
        if vis_cols is not None:
            if col not in vis_cols:
                continue
        
        ax = fig.add_subplot(gs[i,0])
        hist = total_state[:,i]
        
        # background color setting
        clr = plt.cm.Purples(0.9)
        ax.set_facecolor(plt.cm.Blues(0.2))
        
        # target line
        if col in target_cols:
            ax.axhline(targets_dict[col], xmin = 0, xmax = 1, linewidth = 4, color = 'y')
        
        # original data
        hist_real = df_shot[col].values
        t_axis = [dt * j for j in range(min(len(hist), len(hist_real)))]
        ax.plot(t_axis, hist_real[:len(t_axis)], 'k', label = "{}-real".format(col2str[col]))
        
        # RL control data
        ax.plot(t_axis, hist[:len(t_axis)], 'r', label = "{}-controlled".format(col2str[col]))
            
        # axis line for separating control regime
        t_control = seq_len * dt
        ax.axvline(t_control, ymin = 0, ymax = 1, linewidth = 2, color = 'b')
        
        # label and legend
        if i == len(total_cols) - 1:
            ax.set_xlabel("time(s)")
        else:
            plt.setp(ax, xticklabels=[])
            
        ax.set_ylabel(col2str[col])
        ax.legend(loc = "upper right")
        
    # reward
    ax = fig.add_subplot(gs[:,1])
    t_axis = [dt * (i+1) for i in range(len(reward_list))]
    
    clr = plt.cm.Purples(0.9)
    ax.set_facecolor(plt.cm.Blues(0.2))
    ax.plot(t_axis, reward_list, label = 'reward', color = clr)
    
    ax.set_xlabel('time')
    ax.set_ylabel("reward")
    ax.legend(loc = 'upper right')
    
    if env.shape_predictor is not None:
        # plot the profile
        ax = fig.add_subplot(gs[:,2])
        ax.set_title("PINN test result : $\Psi$, $J_\phi$, $P(\psi)$ profile")
        R = env.shape_predictor.R2D
        Z = env.shape_predictor.Z2D
        psi = env.flux.squeeze(0).detach().cpu().numpy()
        ax.contourf(R,Z, psi, levels = 32)
        
        ax = draw_KSTAR_limiter(ax)
        
        '''
        try:
            (r_axis, z_axis), _ = env.shape_predictor.find_axis(env.flux, eps = 1e-3)
            xpts = env.shape_predictor.find_xpoints(env.flux, eps = 1e-3)
            
        except:
            r_axis = None
            z_axis = None
            xpts = []
        
        if r_axis is not None:
            ax.plot(r_axis, z_axis, "o", c = "r", label = "magnetic axis", linewidth = 2)
            ax.legend(loc = 'upper right')
            
        if len(xpts) > 0:
            r_xpts = []
            z_xpts = []
            psi_xpts = []
            
            for r_xpt, z_xpt, psi_xpt in xpts:
                r_xpts.append(r_xpt)
                z_xpts.append(z_xpt)
                psi_xpts.append(psi_xpt)
            
            r_xpts = np.array(r_xpts)
            z_xpts = np.array(z_xpts)
            psi_xpts = np.array(psi_xpts)
            # psi_b = np.min(psi_xpts)
            psi_b = 0.08
        
        else:
            psi_b = 0.08
            
        try:
            if len(xpts) > 0:
                contours = measure.find_contours(psi, psi_b)
                dist_list = []
                for contour in contours:
                    r_contour = R.min() + (R.max() - R.min()) * contour[:,1] / R.shape[0]
                    z_contour = Z.min() + (Z.max() - Z.min()) * contour[:,0] / Z.shape[0]
                    dist = np.mean((r_contour-r_axis) ** 2 + (z_contour - z_axis) ** 2)
                    dist_list.append(dist)
                    
                b_contour = contours[np.argmin(np.array(dist_list))]
                
            else:
                b_contour = None
        except:
            b_contour = None
                    
        if b_contour is not None:
            r_contour = R.min() + (R.max() - R.min()) * b_contour[:,1] / R.shape[0]
            z_contour = Z.min() + (Z.max() - Z.min()) * b_contour[:,0] / Z.shape[0]
            ax.plot(r_contour, z_contour, c = 'r', linewidth = 2)
        '''
            
        # new version : use contour regressor
        if env.contour_regressor is not None:
            
            contour = env.contour
            ax.plot(contour[:,0], contour[:,1], c = 'r', linewidth = 2)
            
            axis = env.axis
            ax.plot(axis[0], axis[1], "o", c = "r", label = "magnetic axis", linewidth = 2)
            ax.legend(loc = 'upper right')
        
        norm = colors.Normalize(vmin = psi.min(), vmax = psi.max())
        map = cm.ScalarMappable(norm=norm)
        fig.colorbar(map)
        ax.set_xlabel("R[m]")
        ax.set_ylabel("Z[m]")
        ax.set_title('Poloidal flux ($\psi$)')
    
    fig.tight_layout()
    plt.savefig(save_file)
    
    # control value plot
    title = "{}_shot_{}_operation_control".format(tag, shot_num)
    save_file = os.path.join(save_dir, "{}.png".format(title))
    
    fig, axes = plt.subplots(len(cols_control)//2, 2, figsize = (16,10), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for idx, (ax, col) in enumerate(zip(axes.ravel(), cols_control)):
        
        hist = total_action[:,idx]
        
        # background color setting
        clr = plt.cm.Purples(0.9)
        ax.set_facecolor(plt.cm.Blues(0.2))
        
        # original data
        hist_real = df_shot[col].values
        t_axis = [dt * j for j in range(min(len(hist), len(hist_real)))]
        ax.plot(t_axis, hist_real[:len(t_axis)], 'k', label = "{}-real".format(col2str[col]))
        
        # RL control data
        ax.plot(t_axis, hist[:len(t_axis)], 'r', label = "{}-controlled".format(col2str[col]))
        
        # target line
        if col in target_cols:
            ax.axhline(targets_dict[col], xmin = 0, xmax = 1, linewidth = 4, color = 'y')
            
        # axis line for separating control regime
        t_control = seq_len * dt
        ax.axvline(t_control, ymin = 0, ymax = 1, linewidth = 2, color = 'b')
        
        # label and legend
        if idx >= len(axes.ravel()) - 2:
            ax.set_xlabel("time(s)")
        else:
            plt.setp(ax, xticklabels=[])
            
        ax.set_ylabel(col2str[col])
        ax.legend(loc = "upper right")
        
    fig.tight_layout()
    plt.savefig(save_file)
    
    return total_state, total_action