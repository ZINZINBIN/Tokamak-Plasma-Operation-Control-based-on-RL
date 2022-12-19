import pandas as pd
import numpy as np
from typing import List
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

def ts_interpolate(df : pd.DataFrame, cols : List, dt : float = 1.0 / 210):
    
    df_interpolate = pd.DataFrame()
    shot_list = np.unique(df.shot.values).tolist()
    
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
            dict_extend[col] = data_extend

        df_shot_extend = pd.DataFrame(data = dict_extend)
        df_interpolate = pd.concat([df_interpolate, df_shot_extend], axis = 0).reset_index(drop = True)

    return df_interpolate

if __name__ == "__main__":
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data.csv")
    cols = df.columns[df.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    fps = 210
    dt = 1.0 / fps * 4

    df_extend = ts_interpolate(df, cols, dt)
    df_extend['frame_idx'] = df_extend.time.apply(lambda x : int(round(x * fps)))
    df_extend.to_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv", index = False)