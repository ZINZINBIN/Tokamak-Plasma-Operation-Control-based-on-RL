import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Literal
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

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