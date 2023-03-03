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

def generate_shot_data_from_real(
    model : torch.nn.Module,
    df_shot_origin : pd.DataFrame,
    seq_len_0D : int,
    seq_len_ctrl : int,
    pred_len_0D : int,
    cols_0D : List,
    cols_ctrl : List,
    scaler_0D = None,
    scaler_ctrl = None,
    device : str = 'cpu',
    title : str = "",
    save_dir : str = "./result/nn_env_performance.png"
    ):
    
    df_shot = df_shot_origin.copy(deep = True)
    
    if scaler_0D:
        df_shot[cols_0D] = scaler_0D.transform(df_shot[cols_0D].values)
    
    if scaler_ctrl:
        df_shot[cols_ctrl] = scaler_ctrl.transform(df_shot[cols_ctrl].values)
    
    time_x = df_shot['time']
    
    data_0D = df_shot[cols_0D]
    data_ctrl = df_shot[cols_ctrl]
    
    predictions = []
    
    idx = 0
    time_length = idx + seq_len_0D
    idx_max = len(data_0D) - pred_len_0D - seq_len_0D
    
    model.to(device)
    model.eval()
    
    while(idx < idx_max):
        with torch.no_grad():
            input_0D = torch.from_numpy(data_0D.loc[idx+1:idx+seq_len_0D].values).unsqueeze(0)
            input_ctrl = torch.from_numpy(data_ctrl.loc[idx+1:idx+seq_len_ctrl].values).unsqueeze(0)
            
            outputs = model(input_0D.to(device), input_ctrl.to(device)).squeeze(0).cpu().numpy()
            predictions.append(outputs)
            
        time_length += pred_len_0D
        idx = time_length - seq_len_0D
    
    predictions = np.concatenate(predictions, axis = 0)
        
    time_x = time_x.loc[seq_len_0D + 1: seq_len_0D + len(predictions)].values
    actual = data_0D[cols_0D].loc[seq_len_0D + 1 : seq_len_0D + len(predictions)].values
    
    fig, axes = plt.subplots(len(cols_0D)//3, 3, figsize = (16,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    if scaler_0D:
        predictions = scaler_0D.inverse_transform(predictions)
        actual = scaler_0D.inverse_transform(actual)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols_0D)):
        ax.plot(time_x, actual[:,i], 'k', label = "actual")
        ax.plot(time_x, predictions[:,i], 'b', label = "pred")
        ax.set_ylabel(col2str[col])
        ax.legend(loc = "upper right")

    fig.tight_layout()
    plt.savefig(save_dir)

def generate_shot_data_from_self(
    model : torch.nn.Module,
    df_shot_origin : pd.DataFrame,
    seq_len_0D : int,
    seq_len_ctrl : int,
    pred_len_0D : int,
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
    idx_max = len(data_0D) - pred_len_0D - seq_len_0D
    
    model.to(device)
    model.eval()
    
    previous_state = torch.Tensor([])
    next_state = None
    state_list = None
    
    while(idx < idx_max):
        with torch.no_grad():
            if idx == 0:
                input_0D = torch.from_numpy(data_0D.loc[idx+1:idx+seq_len_0D].values).unsqueeze(0)
                input_ctrl = torch.from_numpy(data_ctrl.loc[idx+1:idx+seq_len_ctrl].values).unsqueeze(0)
                state_list = torch.from_numpy(data_0D.loc[idx+1:idx+seq_len_ctrl].values)
                
            else:
                input_0D = previous_state.unsqueeze(0)
                input_ctrl = torch.from_numpy(data_ctrl.loc[idx+1:idx+seq_len_ctrl].values).unsqueeze(0)
                            
            next_state = model(input_0D.to(device), input_ctrl.to(device)) 
                
        time_length += pred_len_0D
        idx = time_length - seq_len_0D
        
        # update previous state
        state_list = torch.concat([state_list, next_state.cpu().squeeze(0)], axis = 0)
        previous_state = state_list[idx:idx+seq_len_0D,:]
        
        # prediction value update
        prediction = next_state.detach().squeeze(0).cpu().numpy()
        predictions.append(prediction)
            
    predictions = np.concatenate(predictions, axis = 0)
    
    time_x = time_x.loc[seq_len_0D + 1: seq_len_0D + len(predictions)].values
    actual = data_0D[cols_0D].loc[seq_len_0D + 1 : seq_len_0D + len(predictions)].values
    
    fig, axes = plt.subplots(len(cols_0D)//3, 3, figsize = (16,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    if scaler_0D:
        predictions = scaler_0D.inverse_transform(predictions)
        actual = scaler_0D.inverse_transform(actual)
    
    fig, axes = plt.subplots(len(cols_0D)//3, 3, figsize = (16,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols_0D)):
        ax.plot(time_x, actual[:,i], 'k', label = "actual")
        ax.plot(time_x, predictions[:,i], 'b', label = "pred")
        ax.set_ylabel(col2str[col])
        ax.legend(loc = "upper right")

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
    pred_len_0D = test_data.pred_len_0D
    seq_len_ctrl = test_data.seq_len_ctrl
    
    is_shot_valid = False
    
    while(not is_shot_valid):
        
        shot_num = random.choice(shot_list)
        
        df_shot_origin = test_data.ts_data[test_data.ts_data.shot == shot_num].reset_index(drop = True)
        df_shot = df_shot_origin.copy(deep = True)
        
        idx_start = 0
        idx = idx_start
        time_length = idx_start + seq_len_0D
        idx_max = len(df_shot) - pred_len_0D - seq_len_0D
        
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
    idx_max = len(data_0D) - pred_len_0D - seq_len_0D
    
    model.to(device)
    model.eval()
    
    while(idx < idx_max):
        with torch.no_grad():
            input_0D = torch.from_numpy(data_0D.loc[idx+1:idx+seq_len_0D].values).unsqueeze(0)
            input_ctrl = torch.from_numpy(data_ctrl.loc[idx+1:idx+seq_len_ctrl].values).unsqueeze(0)
            
            outputs = model(input_0D.to(device), input_ctrl.to(device)).squeeze(0).cpu().numpy()
            predictions.append(outputs)
            
        time_length += pred_len_0D
        idx = time_length - seq_len_0D
    
    predictions = np.concatenate(predictions, axis = 0)
        
    time_x = time_x.loc[seq_len_0D + 1: seq_len_0D + len(predictions)].values
    actual = data_0D[cols_0D].loc[seq_len_0D + 1 : seq_len_0D + len(predictions)].values
    
    fig, axes = plt.subplots(len(cols_0D)//3, 3, figsize = (16,12), sharex=True, facecolor = 'white')
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