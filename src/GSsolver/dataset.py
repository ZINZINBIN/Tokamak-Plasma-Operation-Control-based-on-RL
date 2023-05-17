import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List

# Control value : PFC coil current
class PINNDataset(Dataset):
    def __init__(self, df:pd.DataFrame, cols_0D : List, cols_PFC : List):
        self.df = df
        self.cols_0D = cols_0D
        self.cols_PFC = cols_PFC
        
    def __getitem__(self, idx : int):
        path = self.df['path'].values[idx]
        
        # psi from efit
        target = np.load(path)['psi']
        target = torch.from_numpy(target).float()
        
        x_param = self.df[self.cols_0D].values[idx].reshape(-1,)
        x_param = torch.from_numpy(x_param).float()
        
        x_PFC = self.df[self.cols_PFC].values[idx].reshape(-1,)
        x_PFC = torch.from_numpy(x_PFC).float()
        
        data = {
            "params":x_param,
            "PFCs" : x_PFC,
        }
        
        return data, target
    
    def __len__(self):
        return len(self.df)