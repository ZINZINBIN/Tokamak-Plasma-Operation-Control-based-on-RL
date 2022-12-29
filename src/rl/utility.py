import numpy as np
import pandas as pd
import torch
from typing import List, Optional

class InitGenerator:
    def __init__(self, df : pd.DataFrame, t_init : float, state_cols : List, control_cols : List, seq_len : int, random : bool = False, shot_num : Optional[int] = None):
        self.df = df
        self.t_init = t_init
        self.state_cols = state_cols
        self.control_cols = control_cols
        self.seq_len = seq_len
        self.shot_num = shot_num
        
        self.shot_list = np.unique(df.shot)
        self.random = random
        
    def get_state(self):
        
        if self.random or self.shot_num is None:
            shot_num = self.shot_list[int(np.random.randint(len(self.shot_list)))]
            df_shot = self.df[self.df.shot == shot_num]
            init_indices = df_shot[df_shot.time >= self.t_init].index[0:self.seq_len].values
            init_state = df_shot[self.state_cols].loc[init_indices].values
            init_action = df_shot[self.control_cols].loc[init_indices].values
            
        else:
            shot_num = self.shot_num
            df_shot = self.df[self.df.shot == shot_num]
            init_indices = df_shot[df_shot.time >= self.t_init].index[0:self.seq_len].values
            init_state = df_shot[self.state_cols].loc[init_indices].values
            init_action = df_shot[self.control_cols].loc[init_indices].values
        
        init_state = torch.from_numpy(init_state)
        init_action = torch.from_numpy(init_action)
            
        return init_state, init_action