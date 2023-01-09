import pandas as pd
import numpy as np
from typing import List
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from src.nn_env.config import Config

config = Config()

def ts_interpolate(df : pd.DataFrame, cols : List, dt : float = 1.0 / 210):
    
    df_interpolate = pd.DataFrame()
    shot_list = np.unique(df.shot.values).tolist()
    
    # Control parameter : Nan -> 0
    df[config.DEFAULT_CONTROL_COLS] = df[config.DEFAULT_CONTROL_COLS].fillna(0)
    
    # TS data : Nan -> 0
    df[config.TS_NE_CORE_COLS] = df[config.TS_NE_CORE_COLS].fillna(0)
    df[config.TS_TE_CORE_COLS] = df[config.TS_TE_CORE_COLS].fillna(0)
    df[config.TS_NE_EDGE_COLS] = df[config.TS_NE_EDGE_COLS].fillna(0)
    df[config.TS_TE_EDGE_COLS] = df[config.TS_TE_EDGE_COLS].fillna(0)
    
    # scaling for Ne
    df[config.TS_NE_CORE_COLS] = df[config.TS_NE_CORE_COLS].apply(lambda x : x / (1e19))
    df[config.TS_NE_EDGE_COLS] = df[config.TS_NE_EDGE_COLS].apply(lambda x : x / (1e19))
    
    # scaling for Te
    df[config.TS_TE_CORE_COLS] = df[config.TS_TE_CORE_COLS].apply(lambda x : x / (1e3))
    df[config.TS_TE_EDGE_COLS] = df[config.TS_TE_EDGE_COLS].apply(lambda x : x / (1e3))
    
    # Diagnose paramter
    cols_dia = [x for x in config.DEFAULT_DIAG if x != 'ne_inter01']
    df[cols_dia] = df[cols_dia].fillna(0)
    
    shot_ignore = []
    for shot in tqdm(shot_list, desc = 'remove the invalid values'):
        df_shot = df[df.shot==shot]
        for col in config.DEFAULT_0D_COLS:
            if np.sum(df_shot[col] == 0) > 0.5 * len(df_shot):
                shot_ignore.append(shot)
                break
    
    shot_list = [x for x in shot_list if x not in shot_ignore]
    
    for shot in tqdm(shot_list, desc = 'interpolation process'):
        
        # ts data with shot number = shot
        df_shot = df[df.shot == shot]
        df_shot.fillna(method = 'ffill')
        
        dict_extend = {}
        t = df_shot.time.values.reshape(-1,)

        t_start = 0
        t_end = max(t)
        t_end += dt

        t_extend = np.arange(t_start, t_end + dt, dt)
        dict_extend['time'] = t_extend
        dict_extend['shot'] = [shot for _ in range(len(t_extend))]

        for col in cols:
            data = df_shot[col].values.reshape(-1,)
            interp = interp1d(t, data, kind = 'cubic', fill_value = 'extrapolate')
            data_extend = interp(t_extend).reshape(-1,)
            
            if col == "\\ipmhd":
                dict_extend[col] = data_extend * (-1)
            else:
                dict_extend[col] = data_extend

        df_shot_extend = pd.DataFrame(data = dict_extend)
        df_interpolate = pd.concat([df_interpolate, df_shot_extend], axis = 0).reset_index(drop = True)
        
    # Feature engineering
    df_interpolate['\\TS_NE_CORE_AVG'] = df_interpolate[config.TS_NE_CORE_COLS].mean(axis = 1)
    df_interpolate['\\TS_NE_EDGE_AVG'] = df_interpolate[config.TS_NE_EDGE_COLS].mean(axis = 1)
    
    df_interpolate['\\TS_TE_CORE_AVG'] = df_interpolate[config.TS_TE_CORE_COLS].mean(axis = 1)
    df_interpolate['\\TS_TE_EDGE_AVG'] = df_interpolate[config.TS_TE_EDGE_COLS].mean(axis = 1)

    return df_interpolate

if __name__ == "__main__":
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data.csv")
    cols = df.columns[df.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    fps = 210
    dt = 1.0 / fps * 4

    df_extend = ts_interpolate(df, cols, dt)
    df_extend['frame_idx'] = df_extend.time.apply(lambda x : int(round(x * fps)))
    df_extend.to_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv", index = False)