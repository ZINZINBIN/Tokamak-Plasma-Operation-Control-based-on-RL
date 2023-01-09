import torch
import pandas as pd
import numpy as np
import random
from typing import Optional, List
import matplotlib.pyplot as plt

# feed-forward : use data given from experiment prior to prediction
def real_time_predict(
    model : torch.nn.Module,
    df_shot : pd.DataFrame,
    seq_len : int, 
    pred_len : int, 
    dist : int, 
    cols : List, 
    pred_cols : List, 
    scaler = None,
    device : str = 'cpu',
    title : str = "",
    save_dir : str = "./result/nn_env_performance.png"
    ):
    
    df_shot_copy = df_shot.copy(deep = True)
    
    if scaler:
        df_shot[cols] = scaler.fit_transform(df_shot[cols].values)
    
    time_x = df_shot['time'].values
    data = df_shot[cols].values
    target = df_shot_copy[pred_cols].values
    predictions = []
    
    idx_start = 64
    idx = idx_start
    time_length = idx_start + seq_len + dist
    idx_max = len(data) - pred_len - seq_len - dist
    
    model.to(device)
    model.eval()
    
    while(idx < idx_max):
        with torch.no_grad():
            inputs = torch.from_numpy(data[idx:idx+seq_len,:]).unsqueeze(0).to(device)
            outputs = model(inputs).squeeze(0).cpu().numpy()
            predictions.append(outputs)
            
        time_length += pred_len
        idx = time_length - dist - seq_len
    
    predictions = np.concatenate(predictions, axis = 0)
    # time_x = time_x[seq_len+dist:seq_len+dist + len(predictions)]
    # actual = target[seq_len+dist:seq_len+dist + len(predictions)]
    
    time_x = time_x[idx_start + seq_len+dist:idx_start + seq_len+dist + len(predictions)]
    actual = target[idx_start + seq_len+dist:idx_start + seq_len+dist + len(predictions)]
    
    fig,axes = plt.subplots(len(pred_cols), figsize = (6,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes, pred_cols)):
        ax.plot(time_x, actual[:,i], 'k', label = "{}-real".format(col))
        ax.plot(time_x, predictions[:,i], 'b', label = "{}-predict".format(col))
        ax.set_ylabel(col)
        ax.legend(loc = "upper right")
   
    ax.set_xlabel('time (unit:s)')
    fig.tight_layout()
    plt.savefig(save_dir)

    
# use multi-step prediction to generate the similar shot data
def generate_shot_data(
    model : torch.nn.Module,
    df_shot : pd.DataFrame,
    seq_len : int, 
    pred_len : int, 
    dist : int, 
    state_cols : List,
    control_cols : Optional[List], 
    device : str = 'cpu',
    title : str = "",
    save_dir : str = "./result/nn_env_performance.png"
    ):
    
    df_shot_copy = df_shot.copy(deep = True)
    time_x = df_shot['time'].values
    
    if control_cols is not None:
        control_data = df_shot[control_cols].values
        initial_state = df_shot[state_cols].values
    else:
        control_data = None
        initial_state = df_shot[state_cols].values
    
    target = df_shot_copy[state_cols].values
    
    predictions = []
    
    idx_start = 64
    idx = idx_start
    time_length = idx_start + seq_len + dist
    idx_max = len(time_x) - pred_len - seq_len - dist
    
    model.to(device)
    model.eval()
    
    previous_state = None
    next_state = None
    state_list = None
    
    while(idx < idx_max):
        with torch.no_grad():
            # Input : (1,T_in,C_in + C_control)
            # previous_state : (1,T_in,C_in)
            # next_state : (1,T_out,C_out)
            # check whether initial input or not
            if idx == idx_start:
                previous_state = torch.from_numpy(initial_state[idx:idx+seq_len,:]).unsqueeze(0).to(device)
                state_list = torch.from_numpy(initial_state[idx:idx+time_length,:]).unsqueeze(0).to(device)

            # input data
            if control_data is None:
                inputs = previous_state
            else:
                control_value = torch.from_numpy(control_data[idx:idx+seq_len,:]).unsqueeze(0).to(device)
                inputs = torch.concat([previous_state, control_value], axis = 2)
                
            next_state = model(inputs)
            
            time_length += pred_len
            idx = time_length - dist - seq_len
            
            # update previous state
            state_list = torch.concat([state_list, next_state], axis = 1)
            previous_state = state_list[:,idx:idx+seq_len,:]
            
            # prediction value update
            prediction = next_state.detach().squeeze(0).cpu().numpy()
            predictions.append(prediction)
            
    predictions = np.concatenate(predictions, axis = 0)
    
    # time_x = time_x[seq_len+dist:seq_len+dist + len(predictions)]
    # actual = target[seq_len+dist:seq_len+dist + len(predictions)]
    
    time_x = time_x[idx_start + seq_len+dist:idx_start + seq_len+dist + len(predictions)]
    actual = target[idx_start + seq_len+dist:idx_start + seq_len+dist + len(predictions)]
    
    fig,axes = plt.subplots(len(state_cols), figsize = (6,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes, state_cols)):
        ax.plot(time_x, actual[:,i], 'k', label = "{}-real".format(col))
        ax.plot(time_x, predictions[:,i], 'b', label = "{}-predict".format(col))
        ax.set_ylabel(col)
        ax.legend(loc = "upper right")
   
    ax.set_xlabel('time (unit:s)')
    fig.tight_layout()
    plt.savefig(save_dir)
    
# for tensorboard
def predict_tensorboard(
    model : torch.nn.Module,
    test_data : torch.utils.data.Dataset,
    device : str = 'cpu',
    ):
    
    shot_list = np.unique(test_data.ts_data.shot.values)
    seq_len = test_data.seq_len
    pred_len = test_data.pred_len
    dist = test_data.dist
    
    is_shot_valid = False
    while(not is_shot_valid):
        
        shot_num = random.choice(shot_list)
        
        cols = test_data.cols
        pred_cols = test_data.pred_cols
        
        df_shot = test_data.ts_data[test_data.ts_data.shot == shot_num].reset_index(drop = True)
        df_shot_copy = df_shot.copy(deep = True)
        
        time_x = df_shot['time'].values
        data = df_shot[cols].values
        target = df_shot_copy[pred_cols].values
        predictions = []
        
        idx_start = int(len(df_shot) * 0.2)
        idx = idx_start
        time_length = idx_start + seq_len + dist
        idx_max = len(data) - pred_len - seq_len - dist
        
        if idx_max < 0 or idx_max < idx_start:
            is_shot_valid = False
        else:
            is_shot_valid = True
    
    model.to(device)
    model.eval()
    
    while(idx < idx_max):
        with torch.no_grad():
            inputs = torch.from_numpy(data[idx:idx+seq_len,:]).unsqueeze(0).to(device)
            outputs = model(inputs).squeeze(0).cpu().numpy()
            predictions.append(outputs)
            
        time_length += pred_len
        idx = time_length - dist - seq_len
    
    predictions = np.concatenate(predictions, axis = 0)
    
    time_x = time_x[idx_start + seq_len+dist:idx_start + seq_len+dist + len(predictions)]
    actual = target[idx_start + seq_len+dist:idx_start + seq_len+dist + len(predictions)]
    
    fig, axes = plt.subplots(len(pred_cols), figsize = (6,12), sharex=True, facecolor = 'white')
    plt.suptitle("shot : {}-running-process".format(shot_num))
    
    for i, (ax, col) in enumerate(zip(axes, pred_cols)):
        ax.plot(time_x, actual[:,i], 'k', label = "{}-real".format(col))
        ax.plot(time_x, predictions[:,i], 'b', label = "{}-predict".format(col))
        ax.set_ylabel(col)
        ax.legend(loc = "upper right")
   
    ax.set_xlabel('time (unit:s)')
    fig.tight_layout()
    
    return fig