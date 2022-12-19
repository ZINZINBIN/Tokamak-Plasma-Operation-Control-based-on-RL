import os
import numpy as np
import pandas as pd
import torch
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Literal

DEFAULT_0D_COLS = [
    '\\q0','\\q95', '\\ipmhd', '\\kappa', 
    '\\tritop', '\\tribot','\\betap','\\betan',
    '\\li', '\\WTOT_DLM03'
]

DEFAULT_DIAG = [
    '\\ne_inter01', '\\ne_tci01', '\\ne_tci02', '\\ne_tci03', '\\ne_tci04', '\\ne_tci05',
]

DEFAULT_CONTROL_COLS = [
    '\\nb11_pnb','\\nb12_pnb','\\nb13_pnb',
    '\\RC01', '\\RC02', '\\RC03',
    '\\VCM01', '\\VCM02', '\\VCM03',
    '\\EC2_PWR', '\\EC3_PWR', 
    '\\ECSEC2TZRTN', '\\ECSEC3TZRTN',
    '\\LV01'
]

class DatasetFor0D(Dataset):
    def __init__(
        self, 
        ts_data : pd.DataFrame, 
        seq_len : int = 21, 
        pred_len : int = 1, 
        dist:int = 3, 
        state_cols : List = DEFAULT_0D_COLS,
        control_cols : List = DEFAULT_CONTROL_COLS,
        pred_cols : List = DEFAULT_0D_COLS, 
        interval : int = 14,
        scaler = None,
        ):
        self.ts_data = ts_data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.cols = state_cols + control_cols
        self.state_cols = state_cols
        self.control_cols = control_cols
        self.pred_cols = pred_cols
        
        self.dist = dist
        self.interval = interval

        self.input_indices = []
        self.target_indices = []
        
        self.scaler = scaler
        
        # preprocessing
        self.preprocessing()
        
        # data - label index generation
        self._generate_index()
        
    def preprocessing(self):
    
        # control value : NAN -> 0
        self.ts_data[self.control_cols].fillna(0)
        
        # 0D parameter : NAN -> forward fill
        self.ts_data[self.state_cols].fillna(method='ffill')

    def _generate_index(self):
        shot_list = np.unique(self.ts_data.shot.values).tolist()
        
        # ignore shot which have too many nan values
        shot_ignore = []
        for shot in tqdm(shot_list, desc = 'extract the null data'):
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols].isna().sum()
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break
          
        shot_list = [shot_num for shot_num in shot_list if shot_num not in shot_ignore]

        # preprocessing
        for shot in tqdm(shot_list, desc = "Dataset preprocessing..."):
            
            df_shot = self.ts_data[self.ts_data.shot == shot]
            input_indices = []
            target_indices = []

            idx = 0
            idx_last = len(df_shot.index) - self.seq_len - self.pred_len - self.dist

            while(idx < idx_last):
                row = df_shot.iloc[idx]
                t = row['time']
                
                input_indx = df_shot.index.values[idx]
                target_indx = df_shot.index.values[idx + self.seq_len + self.dist]
                
                input_indices.append(input_indx)
                target_indices.append(target_indx)
                
                if idx_last - idx - self.seq_len - self.dist - self.pred_len < 0:
                    break
                else:
                    idx += self.interval

            self.input_indices.extend(input_indices)
            self.target_indices.extend(target_indices)

    def __getitem__(self, idx:int):
        
        input_idx = self.input_indices[idx]
        target_idx = self.target_indices[idx]
        
        data = self.ts_data[self.state_cols + self.control_cols].loc[input_idx:input_idx + self.seq_len -1].values
        target = self.ts_data[self.pred_cols].loc[target_idx : target_idx + self.pred_len -1].values
        
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()

        return data, target

    def __len__(self):
        return len(self.input_indices)