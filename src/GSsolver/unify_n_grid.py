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
    
    rzbdys = data['rzbdys']
    
    
    is_edit = False
    
    if psi.shape[0] != 65:
        print("path : {} psi interpolation needed".format(path))
        interp_fn = interp2d(R,Z, psi, kind = 'linear', fill_value = None)
        
        R_new = np.linspace(R.min(), R.max(), 65, endpoint = True)
        Z_new = np.linspace(R.min(), R.max(), 65, endpoint = True)
        
        RR, ZZ = np.meshgrid(R_new, Z_new)
        psi = interp_fn(R_new, Z_new).reshape(65, 65)
        is_edit = True
    
    if len(rzbdys) != 256:
        print("path : {} contour interpolation needed".format(path))
        continue
    
    if is_edit:
        np.savez(
            path, 
            psi = psi, 
            R = RR, 
            Z = ZZ,
            aspect = data['aspect'],
            bcentr = data['bcentr'],
            ip = data['ip'],
            rmag = data['rmag'],
            zmag = data['zmag'],
            psi_a = data['psi_a'],
            rzbdys = data['rzbdys'],
            k = data['k'],
            triu = data['triu'],
            tril = data['tril'],
            Rc = data['rcen'],
            a = data['amin']
        )