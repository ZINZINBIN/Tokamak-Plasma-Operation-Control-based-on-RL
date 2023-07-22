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

        self.paths = df['path'].values
        self.params = df[cols_0D].values
        self.PFCs = df[cols_PFC].values
        
        self.df['\\ipmhd'] = self.df['\\ipmhd'].apply(lambda x : x * 10 ** (-6))
        self.df[cols_PFC] = self.df[cols_PFC].apply(lambda x : x * 10 ** (-3))
        
    def __getitem__(self, idx : int):
        path = self.paths[idx]
        
        gfile = np.load(path)
        
        # psi from efit
        target = gfile['psi']
        target = torch.from_numpy(target).float()
        rzbdys = torch.from_numpy(gfile['rzbdys']).float()
        k = torch.from_numpy(gfile['k'].reshape(-1,)).float()
        triu = torch.from_numpy(gfile['triu'].reshape(-1,)).float()
        tril = torch.from_numpy(gfile['tril'].reshape(-1,)).float()
        Rc = torch.from_numpy(gfile['Rc'].reshape(-1,)).float()
        a = torch.from_numpy(gfile['a'].reshape(-1,)).float()
        psi_a = torch.from_numpy(gfile['psi_a'].reshape(-1,)).float()
        
        rad, (rc, zc) = self.compute_polar_coordinate(gfile['rzbdys'])
        rad = torch.from_numpy(rad).float()
        center = torch.Tensor([rc, zc]).float()
        
        x_param = self.df[self.cols_0D].values[idx].reshape(-1,)
        x_param = torch.from_numpy(x_param).float()
        
        x_PFC = self.df[self.cols_PFC].values[idx].reshape(-1,)
        x_PFC = torch.from_numpy(x_PFC).float()
        
        Ip = self.df['\\ipmhd'].values[idx].reshape(-1,)
        Ip = torch.from_numpy(Ip).float()
        
        betap = self.df['\\betap'].values[idx].reshape(-1,)
        betap = torch.from_numpy(betap).float()
        
        data = {
            "params":x_param,
            "PFCs" : x_PFC,
            "Ip" : Ip,
            "betap" : betap,
            "k" : k,
            "triu" : triu,
            "tril": tril,
            "Rc" : Rc,
            "a" : a,
            "psi_a" : psi_a,
            "rzbdys" : rzbdys,
            "rad" : rad,
            "center" : center,
        }
        
        return data, target
    
    def __len__(self):
        return len(self.df)
    
    def compute_polar_coordinate(self, rzbdys):
        rc = 0.5 * (min(rzbdys[:,0]) + max(rzbdys[:,0]))
        ind = rzbdys[:,0].argmax()
        zc = rzbdys[ind, 1]
        r = np.sqrt((rzbdys[:,0] - rc) ** 2 + (rzbdys[:,1] - zc) ** 2)
        return r, (rc, zc)
    
    def generate_mask(self, rzbdys, R, Z):
        
        mask = np.zeros_like(R)
    
        for idx in range(len(rzbdys)):
            x = rzbdys[idx, 0]
            y = rzbdys[idx, 1]
    
            r_idx = np.argmin((R[0,:]-x) ** 2)
            z_idx = np.argmin((Z[:,0]-y) ** 2)
            mask[z_idx, r_idx] = 1
        
        return mask