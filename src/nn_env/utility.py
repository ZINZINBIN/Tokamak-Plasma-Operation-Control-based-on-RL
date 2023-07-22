from sklearn.model_selection import train_test_split
import random, torch, os
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from typing import Literal, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# For reproduction
def seed_everything(seed : int = 42, deterministic : bool = False):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        
def preparing_0D_dataset(
    df : pd.DataFrame,
    cols_0D : List,
    cols_ctrl : List,
    scaler : Literal['Robust', 'Standard', 'MinMax'] = 'Robust'
    ):

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')
    df = df[df.shot > 19000]
    ts_cols = cols_0D + cols_ctrl

    # float type
    for col in ts_cols:
        df[col] = df[col].astype(np.float32)
    
    # shot sampling
    shot_list = np.unique(df.shot.values)
    shot_list_tmp = []
    
    for shot in shot_list:
        df_shot = df[df.shot == shot]
        t_start = df_shot.time.iloc[0]
        t_end = df_shot.time.iloc[-1]   

        if t_end - t_start > 10.0:
            shot_list_tmp.append(shot)
     
    shot_list = shot_list_tmp
    
    print("# of shot : {}".format(len(shot_list)))
    
    # train / valid / test data split
    from sklearn.model_selection import train_test_split
    
    shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = 42)
    shot_train, shot_valid = train_test_split(shot_train, test_size = 0.25, random_state = 42)
    
    df_train = df[df.shot.isin(shot_train)]
    df_valid = df[df.shot.isin(shot_valid)]
    df_test = df[df.shot.isin(shot_test)]
    
    if scaler == 'Standard':
        scaler_0D = StandardScaler()
        scaler_ctrl = StandardScaler()
    elif scaler == 'Robust':
        scaler_0D = RobustScaler()
        scaler_ctrl = RobustScaler()
    elif scaler == 'MinMax':
        scaler_0D = MinMaxScaler()
        scaler_ctrl = MinMaxScaler()
  
    # scaler training
    scaler_0D.fit(df_train[cols_0D].values)
    scaler_ctrl.fit(df_train[cols_ctrl].values)
        
    return df_train, df_valid, df_test, scaler_0D, scaler_ctrl

# get range of each output
def get_range_of_output(df : pd.DataFrame, cols_0D : List):
    
    range_info = {}

    for col in cols_0D:
        min_val = df[col].min()
        max_val = df[col].max()
        range_info[col] = [min_val, max_val]
    
    return range_info