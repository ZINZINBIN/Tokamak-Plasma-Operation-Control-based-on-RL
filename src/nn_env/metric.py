
import numpy as np
from typing import Union, List, Optional
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def MSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = True)

def RMSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = False)

def MAE(gt: np.array, pt: np.array):
    return np.mean(np.abs((gt - pt)))

def R2(gt : np.array, pt : np.array):
    return r2_score(gt, pt)

def compute_metrics(gt : Union[np.ndarray, List], pt : Union[np.ndarray, List], algorithm : Optional[str] = None, is_print : bool = True):
    
    if gt.ndim == 3:
        gt = gt.reshape(-1, gt.shape[2])
        pt = pt.reshape(-1, pt.shape[2])
    
    mse = MSE(gt, pt)
    rmse = RMSE(gt, pt)
    mae = MAE(gt, pt)
    r2 = R2(gt, pt)
    
    if is_print:
        if algorithm:
            print("| {} | mse : {:.3f} | rmse : {:.3f} | mae : {:.3f} | r2-score : {:.3f}".format(algorithm, mse, rmse, mae, r2))
        else:
            print("| mse : {:.3f} | rmse : {:.3f} | mae : {:.3f} | r2-score : {:.3f}".format(mse, rmse, mae, r2))
            
    return mse, rmse, mae, r2