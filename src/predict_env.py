import torch
import pandas as pd
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt

# without feed-forward : use only the value that the model predict itself
def real_time_predict(
    model : torch.nn.Module,
    df_shot : pd.DataFrame,
    seq_len : int, 
    pred_len : int, 
    dist : int, 
    cols : List, 
    pred_cols : List, 
    interval : int, 
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
    target = df_shot[pred_cols].values
    predictions = []
    
    idx = 0
    time_length = seq_len + dist
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
    time_x = time_x[seq_len+dist:seq_len+dist + len(predictions)]
    actual = target[seq_len+dist:seq_len+dist + len(predictions)]
    
    fig,axes = plt.subplots(len(cols), figsize = (6,12), sharex=True, facecolor = 'white')
    plt.suptitle(title)
    
    for i, (ax, col) in enumerate(zip(axes, pred_cols)):
        ax.plot(time_x, actual[:,i], 'k', label = "{}-real".format(col))
        ax.plot(time_x, predictions[:,i], 'b-', label = "{}-predict".format(col))
        ax.set_ylabel(col)
        ax.legend(loc = "upper right")
   
    ax.set_xlabel('time (unit:s)')
    fig.tight_layout()
    plt.savefig(save_dir)