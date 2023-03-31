import torch
import pandas as pd
import numpy as np
import random
from typing import Optional, List
import matplotlib.pyplot as plt
from src.nn_env.dataset import DatasetFor0D

col2str = {
    '\\q0' : 'q0',
    '\\q95' : 'q95', 
    '\\ipmhd' : 'Ip', 
    '\\kappa' : 'K', 
    '\\tritop' : 'tri-top', 
    '\\tribot': 'tri-bot',
    '\\betap': 'betap',
    '\\betan': 'betan',
    '\\li': 'li', 
    '\\WTOT_DLM03' : 'W-tot', 
    '\\ne_inter01' : 'Ne',
    '\\TS_NE_CORE_AVG' : 'Ne-core', 
    '\\TS_TE_CORE_AVG' : 'Te-core'
}
    
def generate_shot_data_from_self(
    model : torch.nn.Module,
    df_shot_origin : pd.DataFrame,
    seq_len_0D : int,
    seq_len_ctrl : int,
    cols_0D : List,
    cols_ctrl : List,
    scaler_0D = None,
    scaler_ctrl = None,
    device : str = 'cpu',
    title : str = "",
    save_dir : str = "./result/nn_env_performance.png"
    ):
    
    df_shot = df_shot_origin.copy(deep = True)
    time_x = df_shot['time']
    
    
    if scaler_0D:
        df_shot[cols_0D] = scaler_0D.transform(df_shot[cols_0D].values)
    
    if scaler_ctrl:
        df_shot[cols_ctrl] = scaler_ctrl.transform(df_shot[cols_ctrl].values)
    
    
    data_0D = df_shot[cols_0D]
    data_ctrl = df_shot[cols_ctrl]
    
    predictions = []
    
    idx = 0
    time_length = idx + seq_len_0D
    idx_max = len(data_0D) - seq_len_0D - 1
    
    model.to(device)
    model.eval()
    
    next_state = None
    
    input_0D = torch.from_numpy(data_0D.loc[idx+1:idx+seq_len_0D].values).unsqueeze(0)
    input_ctrl = torch.from_numpy(data_ctrl.loc[idx+1:idx+seq_len_ctrl].values).unsqueeze(0)
    
    target_0D = torch.from_numpy(data_0D.loc[idx+seq_len_0D].values.reshape(1,1,len(cols_0D)))
    target_ctrl = torch.from_numpy(data_ctrl.loc[idx+seq_len_ctrl].values).reshape(1,1,len(cols_ctrl))   
    
    while(idx < idx_max):
        with torch.no_grad():
         
            next_state = model(input_0D.to(device), input_ctrl.to(device), target_0D.to(device), target_ctrl.to(device))[:,-1,:].view(1,1,len(cols_0D)).detach().cpu()
            
            target_0D = torch.concat((target_0D, next_state), axis = 1)
            target_ctrl = torch.concat((target_ctrl, torch.from_numpy(data_ctrl.loc[idx+1+seq_len_ctrl].values).view(1,1,len(cols_ctrl))), axis = 1)
            
        time_length += 1
        idx = time_length - seq_len_0D
        
    predictions = model(input_0D.to(device), input_ctrl.to(device), target_0D.to(device), target_ctrl.to(device)).view(-1,len(cols_0D)).detach().cpu().numpy()

    time_x_init = time_x.loc[1:seq_len_0D+1].values
    actual_init = data_0D[cols_0D].loc[1:seq_len_0D+1].values
    
    t0 = time_x.loc[seq_len_0D+1]

    time_x = time_x.loc[seq_len_0D: seq_len_0D + len(predictions)].values
    actual = data_0D[cols_0D].loc[seq_len_0D: seq_len_0D + len(predictions)].values
    
    if scaler_0D:
        actual_init = scaler_0D.inverse_transform(actual_init)
        predictions = scaler_0D.inverse_transform(predictions)
        actual = scaler_0D.inverse_transform(actual)
    
    fig, axes = plt.subplots(len(cols_0D), 1, figsize = (10,6), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols_0D)):
        # initial condition
        ax.plot(time_x_init, actual_init[:,i],'k')
        
        # pred vs actual
        ax.plot(time_x, actual[:,i], 'k', label = "actual")
        ax.plot(time_x, predictions[:,i], 'b', label = "pred")
        ax.set_ylabel(col2str[col])
        ax.legend(loc = "upper right")
        
        ax.axvline(t0, ymin = 0, ymax = 1, linewidth = 2, color = 'r')

    fig.tight_layout()
    plt.savefig(save_dir)

# for tensorboard
def predict_tensorboard(
    model : torch.nn.Module,
    test_data : DatasetFor0D,
    device : str = 'cpu',
    ):
    
    shot_list = np.unique(test_data.ts_data.shot.values)
    
    seq_len_0D = test_data.seq_len_0D
    seq_len_ctrl = test_data.seq_len_ctrl
    
    is_shot_valid = False
    
    while(not is_shot_valid):
        
        shot_num = random.choice(shot_list)
        
        df_shot_origin = test_data.ts_data[test_data.ts_data.shot == shot_num].reset_index(drop = True)
        df_shot = df_shot_origin.copy(deep = True)
        
        idx_start = 0
        idx = idx_start
        time_length = idx_start + seq_len_0D
        idx_max = len(df_shot) - seq_len_0D
        
        if idx_max < 0 or idx_max < 32:
            is_shot_valid = False
        else:
            is_shot_valid = True
    
    model.to(device)
    model.eval()
    
    time_x = df_shot['time']
    
    cols_0D = test_data.cols_0D
    cols_ctrl = test_data.cols_ctrl
    
    data_0D = df_shot[cols_0D]
    data_ctrl = df_shot[cols_ctrl]
    
    predictions = []
    
    idx = 0
    time_length = idx + seq_len_0D
    idx_max = len(data_0D) - seq_len_0D - 1
    
    model.to(device)
    model.eval() 

    next_state = None
    
    input_0D = torch.from_numpy(data_0D.loc[idx+1:idx+seq_len_0D].values).unsqueeze(0)
    input_ctrl = torch.from_numpy(data_ctrl.loc[idx+1:idx+seq_len_ctrl].values).unsqueeze(0)
    
    target_0D = torch.from_numpy(data_0D.loc[idx+seq_len_0D].values.reshape(1,1,len(cols_0D)))
    target_ctrl = torch.from_numpy(data_ctrl.loc[idx+seq_len_ctrl].values).reshape(1,1,len(cols_ctrl))   
    
    while(idx < idx_max):
        with torch.no_grad():         
            next_state = model(input_0D.to(device), input_ctrl.to(device), target_0D.to(device), target_ctrl.to(device))[:,-1,:].view(1,1,len(cols_0D)).detach().cpu()
            
            target_0D = torch.concat((target_0D, next_state), axis = 1)
            target_ctrl = torch.concat((target_ctrl, torch.from_numpy(data_ctrl.loc[idx+1+seq_len_ctrl].values).view(1,1,len(cols_ctrl))), axis = 1)
            
        time_length += 1
        idx = time_length - seq_len_0D
    
    predictions = model(input_0D.to(device), input_ctrl.to(device), target_0D.to(device), target_ctrl.to(device)).view(-1,len(cols_0D)).detach().cpu().numpy()
            
    time_x = time_x.loc[seq_len_0D: seq_len_0D + len(predictions)].values
    actual = data_0D[cols_0D].loc[seq_len_0D : seq_len_0D + len(predictions)].values
    
    fig, axes = plt.subplots(len(cols_0D), 1, figsize = (16,10), sharex=True, facecolor = 'white')
    plt.suptitle("shot : {}-running-process".format(shot_num))
    
    if test_data.scaler_0D:
        predictions = test_data.scaler_0D.inverse_transform(predictions)
        actual = test_data.scaler_0D.inverse_transform(actual)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols_0D)):
        ax.plot(time_x, actual[:,i], 'k', label = "actual")
        ax.plot(time_x, predictions[:,i], 'b', label = "pred")
        ax.set_ylabel(col2str[col])
        ax.legend(loc = "upper right")

    fig.tight_layout()
    
    return fig