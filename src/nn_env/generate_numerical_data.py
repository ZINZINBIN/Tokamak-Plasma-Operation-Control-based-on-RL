import pandas as pd
import numpy as np
from typing import List
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from src.nn_env.config import Config
import warnings

warnings.filterwarnings(action = 'ignore')

config = Config()

def ts_interpolate(df : pd.DataFrame, df_disruption : pd.DataFrame, cols : List, dt : float = 1.0 / 210, ewm_interval : int = 8, exist_profile : bool = False):
    
    df_interpolate = pd.DataFrame()
    shot_list = np.unique(df.shot.values).tolist()
    
    # inf, -inf to nan
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # control(especaily heating) parameter : Nan -> 0
    df[config.cols_heating] = df[config.cols_heating].fillna(0)
    df[config.cols_control] = df[config.cols_control].fillna(0)
    
    # nan interpolation
    df[config.cols_efit] = df[config.cols_efit].interpolate(method = 'linear', limit_direction = 'forward')
    df[config.cols_diagnose] = df[config.cols_diagnose].interpolate(method = 'linear', limit_direction= 'forward')
    
    if exist_profile:
        df[config.TS_NE_CORE_COLS] = df[config.TS_NE_CORE_COLS].interpolate(method = 'linear', limit_direction = 'forward')
        df[config.TS_TE_CORE_COLS] = df[config.TS_TE_CORE_COLS].interpolate(method = 'linear', limit_direction = 'forward')
        df[config.TS_NE_EDGE_COLS] = df[config.TS_NE_EDGE_COLS].interpolate(method = 'linear', limit_direction = 'forward')
        df[config.TS_TE_EDGE_COLS] = df[config.TS_TE_EDGE_COLS].interpolate(method = 'linear', limit_direction = 'forward')
    
    if exist_profile:
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
        
    def _bound(x, value : float):
        return x if abs(x) < value else value * x / abs(x)
    
    # Some negative values are mis-estimated due to equipmental problem
    # These values will be replaced by preprocessing
    if '\\ipmhd' in config.cols_efit:
        df['\\ipmhd'] = df['\\ipmhd'].abs().values
        df['\\ipmhd'] = df['\\ipmhd'].apply(lambda x : x / 1e6)
        
    if '\\betap' in config.cols_efit:
        df['\\betap'] = df['\\betap'].apply(lambda x : x if x > 0 else 0)
    
    if '\\betan' in config.cols_efit:
        df['\\betan'] = df['\\betan'].apply(lambda x : x if x > 0 else 0)
        
    # electron density diagnose paramter -> remove negative value and rescale
    if '\\ne_inter01' in config.cols_diagnose:
        df['\\ne_inter01'] = df['\\ne_inter01'].apply(lambda x : x if x > 0 else 0)
        df['\\ne_inter01'] = df['\\ne_inter01'].apply(lambda x : x / 1e4)
    
    if '\\ne_tci01' in config.cols_diagnose:
        df['\\ne_tci01'] = df['\\ne_tci01'].apply(lambda x : x if x > 0 else 0)
        df['\\ne_tci01'] = df['\\ne_tci01'].apply(lambda x : x / 1e4)
        
    if '\\ne_tci02' in config.cols_diagnose:
        df['\\ne_tci02'] = df['\\ne_tci02'].apply(lambda x : x if x > 0 else 0)
        df['\\ne_tci02'] = df['\\ne_tci02'].apply(lambda x : x / 1e4)

    if '\\ne_tci03' in config.cols_diagnose:
        df['\\ne_tci03'] = df['\\ne_tci03'].apply(lambda x : x if x > 0 else 0)
        df['\\ne_tci03'] = df['\\ne_tci03'].apply(lambda x : x / 1e4)

    if '\\ne_tci04' in config.cols_diagnose:
        df['\\ne_tci04'] = df['\\ne_tci04'].apply(lambda x : x if x > 0 else 0)
        df['\\ne_tci04'] = df['\\ne_tci04'].apply(lambda x : x / 1e4)

    if '\\ne_tci05' in config.cols_diagnose:
        df['\\ne_tci05'] = df['\\ne_tci05'].apply(lambda x : x if x > 0 else 0)
        df['\\ne_tci05'] = df['\\ne_tci05'].apply(lambda x : x / 1e4)
        
    # PFPC coil current scaling
    for col in config.cols_control:
        df[col] = df[col].apply(lambda x : x / 1e3)
    
    # remove nan value of diagnose parameter
    cols_dia = [x for x in config.cols_diagnose if x != 'ne_inter01']
    df[cols_dia] = df[cols_dia].fillna(0)

    # outlier removal with rule-based mothod
    if exist_profile:
        # Tomson scattering measurement
        for col in config.TS_NE_CORE_COLS:
            df[col] = df[col].apply(lambda x : 0 if abs(x/1e6)>1 else x)
        
        for col in config.TS_NE_EDGE_COLS:
            df[col] = df[col].apply(lambda x : 0 if abs(x/1e6)>1 else x)
        
        for col in config.TS_TE_CORE_COLS:
            df[col] = df[col].apply(lambda x : 0 if abs(x/1e3)>1 else x)
        
        for col in config.TS_TE_EDGE_COLS:
            df[col] = df[col].apply(lambda x : 0 if abs(x/1e3)>1 else x)

    # filtering the experiment : too many nan values for measurement or time length is too short
    shot_ignore = []
    ignore_by_time_length = []
    ignore_by_const_value = []
    ignore_by_null = []
    
    for shot in tqdm(shot_list, desc = 'remove the invalid values'):
        # dataframe per shot
        df_shot = df[df.shot==shot]
        
        # time length of the experiment is too short : at least larger than 2(s)
        if df_shot.time.iloc[-1] - df_shot.time.iloc[0] < 2.0:
            shot_ignore.append(shot)
            ignore_by_time_length.append(shot)
            continue
        
        # measurement error : null data or constant that the measure did not proceed well
        for col in config.cols_efit:
            # null data
            if np.sum(df_shot[col] == 0) > 0.5 * len(df_shot):
                shot_ignore.append(shot)
                ignore_by_null.append(shot)
                break
            
            # constant value
            if df_shot[col].max() - df_shot[col].min() < 1e-4:
                shot_ignore.append(shot)
                ignore_by_const_value.append(shot)
                break
            
    shot_list = [x for x in shot_list if x not in shot_ignore]
    
    print("######### Dataset Information ############")
    print("# of shot ignored : ", len(shot_ignore))
    print("# of shot due to time length : ", len(ignore_by_time_length))
    print("# of shot due to null value : ", len(ignore_by_null))
    print("# of shot due to const value : ", len(ignore_by_const_value))
    print("# of shot for use : ", len(shot_list))
    
    print("\n######### Interpolation process ############")
    for shot in tqdm(shot_list, desc = 'interpolation process'):
        
        # ts data with shot number = shot
        df_shot = df[df.shot == shot]
        df_shot[config.cols_efit] = df_shot[config.cols_efit].fillna(method = 'ffill')
        
        # outlier replacement
        for col in cols:
            
            # plasma current -> pass
            if col == '\\ipmhd':
                continue
            
            q1 = df_shot[col].quantile(0.15)
            q3 = df_shot[col].quantile(0.85)
            
            IQR = q3 - q1
            whisker_width = 1.25      
            
            lower_whisker = q1 - whisker_width * IQR
            upper_whisker = q3 + whisker_width * IQR
            
            df_shot.loc[:,col] = np.where(df_shot[col]>upper_whisker, upper_whisker, np.where(df_shot[col]<lower_whisker,lower_whisker, df_shot[col]))
        
        dict_extend = {}
        t = df_shot.time.values.reshape(-1,)

        # quench info
        if shot in df_disruption.shot.values:
            tTQend = df_disruption[df_disruption.shot == shot].t_tmq.values[0]
            tftsrt = df_disruption[df_disruption.shot == shot].t_flattop_start.values[0]
            
        else:
            tTQend = max(t)
            tftsrt = 1.25
        
        # define t_start and t_end
        t_start = min(t)
        t_end = max(t)
        
        # valid shot selection
        if t_end < tftsrt:
            print("Invalid shot : {} - loss of data".format(shot))
            continue
        elif t_end < 2.0:
            print("Invalid shot : {} - operation time is too short".format(shot))
            continue
        elif int((t_end - tftsrt) // dt) < 4:
            print("Invalid shot : {} - data too small".format(shot))
            continue
        
        # correct the t_end
        # we want to see flattop interval
        t_start = tftsrt
        
        if t_end >= tTQend:
            t_end = tTQend - dt * 4
            
        # double check : n points should be larger than 4 for stable interpolation
        if int((t_end - t_start) // dt) < 4:
            print("Invalid shot : {} - data too small".format(shot))
            continue
        elif t_end < t_start:
            print("Invalid shot : {} - t_end is smaller than t_start".format(shot))
            continue
        elif t_end - t_start < 2.0:
            print("Invalid shot : {} - operation time is too short, t_start : {:.3f} and t_end : {:.3f}".format(shot, t_start, t_end))
            continue

        t_extend = np.arange(t_start, t_end + dt, dt)
        dict_extend['time'] = t_extend
        dict_extend['shot'] = [shot for _ in range(len(t_extend))]

        for col in cols:
            data = df_shot[col].values.reshape(-1,)
            interp = interp1d(t, data, kind = 'linear', fill_value = 'extrapolate')
            data_extend = interp(t_extend).reshape(-1,)
            
            if col == "\\ipmhd":
                dict_extend[col] = np.abs(data_extend)
            else:
                dict_extend[col] = data_extend

        df_shot_extend = pd.DataFrame(data = dict_extend)
        
        # moving average : EWM
        df_shot_extend = df_shot_extend.ewm(ewm_interval).mean()
        df_interpolate = pd.concat([df_interpolate, df_shot_extend], axis = 0).reset_index(drop = True)
    
    # Feature engineering    
    if exist_profile:
        df_interpolate['\\TS_NE_CORE_AVG'] = df_interpolate[config.TS_NE_CORE_COLS].mean(axis = 1)
        df_interpolate['\\TS_NE_EDGE_AVG'] = df_interpolate[config.TS_NE_EDGE_COLS].mean(axis = 1)
        
        df_interpolate['\\TS_TE_CORE_AVG'] = df_interpolate[config.TS_TE_CORE_COLS].mean(axis = 1)
        df_interpolate['\\TS_TE_EDGE_AVG'] = df_interpolate[config.TS_TE_EDGE_COLS].mean(axis = 1)
    
    # convert type of shot
    df_interpolate['shot'] = df_interpolate['shot'].astype(int)
        
    # negative value removal
    if '\\ipmhd' in config.cols_efit:
        df_interpolate['\\ipmhd'] = df_interpolate['\\ipmhd'].abs().values
        
    if '\\betap' in config.cols_efit:
        df_interpolate['\\betap'] = df_interpolate['\\betap'].apply(lambda x : x if x > 0 else 0)
    
    if '\\betan' in config.cols_efit:
        df_interpolate['\\betan'] = df_interpolate['\\betan'].apply(lambda x : x if x > 0 else 0)
        
    if '\\WTOT_DLM03' in config.cols_efit:
        df_interpolate['\\WTOT_DLM03'] = df_interpolate['\\WTOT_DLM03'].apply(lambda x : x if x > 0 else 0)
        
    if '\\li' in config.cols_efit:
        df_interpolate['\\li'] = df_interpolate['\\li'].apply(lambda x : x if x > 0 else 0)
        
    if '\\ne_inter01' in config.cols_diagnose:
        df_interpolate['\\ne_inter01'] = df_interpolate['\\ne_inter01'].apply(lambda x : x if x > 0 else 0)

    return df_interpolate

if __name__ == "__main__":
    df = pd.read_csv("./dataset/KSTAR_rl_control_ts_data_0704.csv")
    df_disrupt = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List_2022.csv", encoding='cp949')
    
    print(df.describe())
    
    cols = df.columns[df.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()
    dt = 0.05
    ewm_interval = 8

    df_extend = ts_interpolate(df, df_disrupt, cols, dt, ewm_interval)
    df_extend.to_csv("./dataset/KSTAR_rl_control_ts_data_extend.csv", index = False)
    
    print(df_extend.describe())