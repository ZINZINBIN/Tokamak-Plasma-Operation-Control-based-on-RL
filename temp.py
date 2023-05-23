import pandas as pd
new_df = pd.read_csv("./dataset/KSTAR_rl_GS_solver.csv")

import numpy as np

npz_path_list = new_df['path']

from scipy.interpolate import interp2d
from tqdm.auto import tqdm

for path in tqdm(npz_path_list):
    data = np.load(path)
    psi = data['psi']
    R = data['R']
    Z = data['Z']
    
    if psi.shape[0] != 65:
        
        interp_fn = interp2d(R,Z, psi, kind = 'linear', fill_value = None)
        
        R_new = np.linspace(R.min(), R.max(), 65, endpoint = True)
        Z_new = np.linspace(R.min(), R.max(), 65, endpoint = True)
        
        RR, ZZ = np.meshgrid(R_new, Z_new)
        psi = interp_fn(R_new, Z_new).reshape(65, 65)
        np.savez(path, psi = psi, R = RR, Z = ZZ)
        
    else:
        continue