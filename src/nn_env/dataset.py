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
    '\\li', '\\WTOT_DLM03', '\\ne_inter01',
    '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
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

# NN based predictor uses 0D parameters and control parameters to predict the next state of the plasma
#   However, considering RL application, control parameters should contains future values 
#   which is equal to future 0D parameters over next time
# Thus, two different input data should be generated

class DatasetFor0D(Dataset):
    def __init__(
        self, 
        ts_data : pd.DataFrame, 
        disrupt_data : pd.DataFrame,
        seq_len_0D : int = 10,
        seq_len_ctrl : int = 11,
        pred_len_0D : int = 1,
        cols_0D : List = DEFAULT_0D_COLS,
        cols_ctrl : List = DEFAULT_CONTROL_COLS,
        interval : int = 3,
        scaler_0D = None,
        scaler_ctrl = None,
        multi_step : bool = False
        ):
        
        self.multi_step = multi_step
        
        # dataframe : time series data and disruption information
        self.ts_data = ts_data
        self.disrupt_data = disrupt_data
        
        # dataset info
        self.seq_len_0D = seq_len_0D
        self.seq_len_ctrl = seq_len_ctrl
        
        self.cols_0D = cols_0D
        self.cols_ctrl = cols_ctrl
        
        self.pred_len_0D = pred_len_0D
        self.interval = interval
        
        # scaler
        self.scaler_0D = scaler_0D
        self.scaler_ctrl = scaler_ctrl
        
        # indice for getitem method
        self.input_indices = []
        self.target_indices = []
        
        # experiment list
        self.shot_list = np.unique(self.ts_data.shot.values).tolist()
        
        # preprocessing
        self.preprocessing()
        
        # data - label index generation
        self._generate_index()
        
    def preprocessing(self):
        
        # control value : NAN -> 0
        self.ts_data[self.cols_ctrl] = self.ts_data[self.cols_ctrl].fillna(0)
        
        # ignore shot which have too many nan values
        shot_ignore = []
        
        for shot in tqdm(self.shot_list, desc = 'extract the null data'):
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols_0D + self.cols_ctrl].isna().sum()
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break
        
        # update shot list with ignoring the null data
        shot_list_new = [shot_num for shot_num in self.shot_list if shot_num not in shot_ignore]
        self.shot_list = shot_list_new
        
        # 0D parameter : NAN -> forward fill
        for shot in tqdm(self.shot_list, desc = 'replace nan value'):
            df_shot = self.ts_data[self.ts_data.shot == shot].copy()
            self.ts_data.loc[self.ts_data.shot == shot, self.cols_0D] = df_shot[self.cols_0D].fillna(method='ffill')
                    
        # scaling
        if self.scaler_0D:
            self.ts_data[self.cols_0D] = self.scaler_0D.transform(self.ts_data[self.cols_0D])
            
        if self.scaler_ctrl:
            self.ts_data[self.cols_ctrl] = self.scaler_ctrl.transform(self.ts_data[self.cols_ctrl])

    def _generate_index(self):
        
        # disruption data : tftsrt, tipminf
        df_disruption = self.disrupt_data

        # Index generation
        for shot in tqdm(self.shot_list, desc = "Dataset Indices generation..."):
            
            if shot not in df_disruption.shot.values:
                tftsrt = 1.25
            else:
                tftsrt = df_disruption[df_disruption.shot == shot].t_flattop_start.values[0]
            
            df_shot = self.ts_data[self.ts_data.shot == shot]
            input_indices = []
            target_indices = []

            idx = 0
            idx_last = len(df_shot.index) - self.seq_len_0D - self.pred_len_0D
            
            if idx_last < 10:
                continue

            while(idx < idx_last):
                row = df_shot.iloc[idx]
                t = row['time']
                
                if t < tftsrt:
                    idx += self.interval
                    continue
                
                input_indx = df_shot.index.values[idx]
                target_indx = df_shot.index.values[idx + self.seq_len_0D]
                
                input_indices.append(input_indx)
                target_indices.append(target_indx)
                
                if idx_last - idx - self.seq_len_ctrl < 0:
                    break
                
                else:
                    idx += self.interval

            self.input_indices.extend(input_indices)
            self.target_indices.extend(target_indices)

    def __getitem__(self, idx:int):
        
        if self.multi_step:
            # second version : multi-step training 
            input_idx = self.input_indices[idx]
            target_idx = self.target_indices[idx]
            
            data_0D = self.ts_data[self.cols_0D].loc[input_idx+1:input_idx + self.seq_len_0D].values
            data_ctrl = self.ts_data[self.cols_ctrl].loc[input_idx+1:input_idx + self.seq_len_ctrl].values
            
            target_0D = self.ts_data[self.cols_0D].loc[target_idx: target_idx + self.pred_len_0D-1].values
            target_ctrl = self.ts_data[self.cols_ctrl].loc[target_idx: target_idx + self.pred_len_0D-1].values
            
            label = self.ts_data[self.cols_0D].loc[target_idx+1: target_idx + self.pred_len_0D].values
            
            data_0D = torch.from_numpy(data_0D).float()
            data_ctrl = torch.from_numpy(data_ctrl).float()

            target_0D = torch.from_numpy(target_0D).float()
            target_ctrl = torch.from_numpy(target_ctrl).float()
            
            label = torch.from_numpy(label).float()

            return data_0D, data_ctrl, target_0D, target_ctrl, label
        
        else:
            # first version : single step training
            input_idx = self.input_indices[idx]
            target_idx = self.target_indices[idx]
            
            data_0D = self.ts_data[self.cols_0D].loc[input_idx:input_idx + self.seq_len_0D-1].values
            data_ctrl = self.ts_data[self.cols_ctrl].loc[input_idx:input_idx + self.seq_len_ctrl-1].values
            
            target = self.ts_data[self.cols_0D].loc[target_idx:target_idx + self.pred_len_0D-1].values
            
            data_0D = torch.from_numpy(data_0D).float()
            data_ctrl = torch.from_numpy(data_ctrl).float()
            target = torch.from_numpy(target).float()

            return data_0D, data_ctrl, target
                
    def __len__(self):
        return len(self.input_indices)

# for validation : multi-step prediction
# in this case, batch size must be 1 due to different input and target size

class DatasetForMultiStepPred(Dataset):
    def __init__(
        self, 
        ts_data : pd.DataFrame, 
        disrupt_data : pd.DataFrame,
        seq_len_0D : int = 20,
        seq_len_ctrl : int = 24,
        pred_len_0D : int = 16,
        cols_0D : List = DEFAULT_0D_COLS,
        cols_ctrl : List = DEFAULT_CONTROL_COLS,
        interval : int = 10,
        scaler_0D = None,
        scaler_ctrl = None,
        ):
        
        # dataframe : time series data and disruption information
        self.ts_data = ts_data
        self.disrupt_data = disrupt_data
        
        # dataset info
        self.seq_len_0D = seq_len_0D
        self.seq_len_ctrl = seq_len_ctrl
        
        self.cols_0D = cols_0D
        self.cols_ctrl = cols_ctrl
        
        # maximum prediction length
        self.pred_len_0D = pred_len_0D
        self.interval = interval
        
        # scaler
        self.scaler_0D = scaler_0D
        self.scaler_ctrl = scaler_ctrl
        
        # indice for getitem method
        self.input_indices = []
        self.target_indices = []
        
        # experiment list
        self.shot_list = np.unique(self.ts_data.shot.values).tolist()
        
        # preprocessing
        self.preprocessing()
        
        # data - label index generation
        self._generate_index()
        
    def preprocessing(self):
        
        # control value : NAN -> 0
        self.ts_data[self.cols_ctrl] = self.ts_data[self.cols_ctrl].fillna(0)
        
        # ignore shot which have too many nan values
        shot_ignore = []
        
        for shot in tqdm(self.shot_list, desc = 'extract the null data'):
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols_0D + self.cols_ctrl].isna().sum()
            
            # null limit
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break
            
            # length limit
            if len(df_shot) < 2 * self.seq_len_0D + self.pred_len_0D * 2:
                shot_ignore.append(shot)
                break
        
        # update shot list with ignoring the null data
        shot_list_new = [shot_num for shot_num in self.shot_list if shot_num not in shot_ignore]
        self.shot_list = shot_list_new
        
        # 0D parameter : NAN -> forward fill
        for shot in tqdm(self.shot_list, desc = 'replace nan value'):
            df_shot = self.ts_data[self.ts_data.shot == shot].copy()
            self.ts_data.loc[self.ts_data.shot == shot, self.cols_0D] = df_shot[self.cols_0D].fillna(method='ffill')
                    
        # scaling
        if self.scaler_0D:
            self.ts_data[self.cols_0D] = self.scaler_0D.transform(self.ts_data[self.cols_0D])
            
        if self.scaler_ctrl:
            self.ts_data[self.cols_ctrl] = self.scaler_ctrl.transform(self.ts_data[self.cols_ctrl])

    def _generate_index(self):
        
        # disruption data : tftsrt, tipminf
        df_disruption = self.disrupt_data

        # Index generation
        for shot in tqdm(self.shot_list, desc = "Dataset Indices generation..."):
            
            if shot not in df_disruption.shot.values:
                tftsrt = 1.25
            else:
                # tftsrt = df_disruption[df_disruption.shot == shot].tftsrt.values[0]
                tftsrt = df_disruption[df_disruption.shot == shot].t_flattop_start.values[0]
                
            df_shot = self.ts_data[self.ts_data.shot == shot]
                
            idx_start = 0
            idx = 0
            idx_last = len(df_shot.index) - self.seq_len_0D - self.pred_len_0D
            
            df_shot = self.ts_data[self.ts_data.shot == shot]
            
            if idx_last < 10:
                continue
            
            # finding idx_start
            while(idx_start < idx_last):
                row = df_shot.iloc[idx]
                t = row['time']
                
                if t < tftsrt:
                    idx += 1
                    continue
                else:
                    idx_start = idx
                    break
                
            input_indx = df_shot.index.values[idx_start]
            target_indx = df_shot.index.values[idx_start + self.seq_len_0D]
            
            # update input index and target index
            self.input_indices.append(input_indx)
            self.target_indices.append(target_indx)
        
    def __getitem__(self, idx:int):
        
        input_idx = self.input_indices[idx]
        target_idx = self.target_indices[idx]
        
        data_0D = self.ts_data[self.cols_0D].loc[input_idx:input_idx + self.seq_len_0D-1].values
        data_ctrl = self.ts_data[self.cols_ctrl].loc[input_idx:input_idx + self.seq_len_0D + self.pred_len_0D].values
        target = self.ts_data[self.cols_0D].loc[target_idx:target_idx + self.pred_len_0D].values
        
        data_0D = torch.from_numpy(data_0D).float()
        data_ctrl = torch.from_numpy(data_ctrl).float()
        target = torch.from_numpy(target).float()

        return data_0D, data_ctrl, target
                
    def __len__(self):
        return len(self.input_indices)
