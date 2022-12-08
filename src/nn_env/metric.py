
import numpy as np
from typing import Union, List, Optional
from sklearn.metrics import mean_squared_error

def MSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = True)

def RMSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = False)

def MAE(gt: np.array, pt: np.array):
    return np.mean(np.abs((gt - pt)))

def compute_metrics(gt : Union[np.ndarray, List], pt : Union[np.ndarray, List], algorithm : Optional[str] = None, is_print : bool = True):
    mse = MSE(gt, pt)
    rmse = RMSE(gt, pt)
    mae = MAE(gt, pt)
    
    if is_print:
        if algorithm:
            print("# {}, mse : {:.3f}, rmse : {:.3f}, mae : {:.3f}".format(algorithm, mse, rmse, mae))
        else:
            print("# mse : {:.3f}, rmse : {:.3f}, mae : {:.3f}".format(mse, rmse, mae))
            
    return mse, rmse, mae