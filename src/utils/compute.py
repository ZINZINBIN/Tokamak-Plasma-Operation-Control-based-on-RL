import numpy as np
from scipy.linalg import lu
from typing import Union, Optional
from src.utils.physical_constant import EPS
import math

def Compute1DIntegral(A_origin : Union[np.array, np.ndarray], x_min : float, x_max : float):
    N = A_origin.shape[0]
    dx = (x_max - x_min) / N
    total = 0
    for idx in range(0,N-1):
        diff = 0.5 * (A_origin[idx] + A_origin[idx + 1]) * dx
        total += diff
        
    return total

def Compute2DIntegral(A_origin, x_min : float, x_max : float, y_min : float, y_max : float):
    
    n_mesh_x = A_origin.shape[0]
    n_mesh_y = A_origin.shape[1]
    
    dx = (x_max - x_min) / n_mesh_x
    dy = (y_max - y_min) / n_mesh_y 
    
    result = 0
    
    for idx_x in range(0, n_mesh_x - 1):
        for idx_y in range(0, n_mesh_y - 1):
            diff = 0.25 * (A_origin[idx_x, idx_y] + A_origin[idx_x + 1, idx_y] + A_origin[idx_x, idx_y + 1] + A_origin[idx_x + 1, idx_y + 1])
            diff *= dx
            diff *= dy
            result += diff
            
    return result
            
def compute_B_r(psi, R_min, R_max, Z_min, Z_max):
    '''compute B_r from poloidal magnetic flux
    - psi : 2D array with poloidal magnetic flux
    - R_min / R_max / Z_min / Z_max : geometrical components
    - method : FDM 
    '''
    
    Br = np.zeros_like(psi)
    
    Nr = psi.shape[0]
    Nz = psi.shape[1]
    
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    
    for idx_r in range(0, Nr):
        for idx_z in range(0,Nz):
            R = R_min + idx_r * dR
            
            if idx_z == 0:
                d_psi = (-1) * psi[idx_r, idx_z + 2] + 4 * psi[idx_r, idx_z + 1] - 3 * psi[idx_r, idx_z]
                Br[idx_r, idx_z] = (-1) * d_psi / 2 / dZ / R
                
            elif idx_z == Nz - 1 :
                d_psi = (-1) * psi[idx_r, idx_z] + 4 * psi[idx_r, idx_z - 1] - 3 * psi[idx_r, idx_z - 2]
                Br[idx_r, idx_z] = (-1) * d_psi / 2 / dZ / R
                
            else:
                d_psi = psi[idx_r, idx_z + 1] - psi[idx_r, idx_z - 1]
                Br[idx_r, idx_z] = d_psi / 2 / dZ
                
    return Br

def compute_B_z(psi, R_min, R_max, Z_min, Z_max):
    '''compute B_z from poloidal magnetic flux
    - psi : 2D array with poloidal magnetic flux
    - R_min / R_max / Z_min / Z_max : geometrical components
    - method : FDM 
    '''
    
    Bz = np.zeros_like(psi)
    
    Nr = psi.shape[0]
    Nz = psi.shape[1]
    
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    
    for idx_r in range(0, Nr):
        for idx_z in range(0,Nz):
            R = R_min + idx_r * dR
            
            if idx_r == 0:
                d_psi = (-1) * psi[idx_r + 2, idx_z + 2] + 4 * psi[idx_r + 1, idx_z] - 3 * psi[idx_r, idx_z]
                Bz[idx_r, idx_z] = d_psi / 2 / dR / R
                
            elif idx_r == Nr - 1 :
                d_psi = (-1) * psi[idx_r, idx_z] + 4 * psi[idx_r - 1, idx_z] - 3 * psi[idx_r - 2, idx_z]
                Bz[idx_r, idx_z] = d_psi / 2 / dR / R
                
            else:
                d_psi = psi[idx_r + 1, idx_z] - psi[idx_r - 1, idx_z]
                Bz[idx_r, idx_z] = d_psi / 2 / dR
                
    return Bz

def compute_B_phi(psi, R_min, R_max, Z_min, Z_max):
    
    B_phi = np.zeros_like(psi)
    Nr = psi.shape[0]
    Nz = psi.shape[1]

    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    
    for idx_r in range(0, Nr):
        for idx_z in range(0, Nz):
            R = R_min + idx_r * dR
            Z = Z_min + idx_z * dZ 
    

    return None

def compute_J_phi_plasma(psi : float, psi_a : float, psi_b : float, r : float, r0 : float, lamda : float, beta_0 : float, m : int, n: int):
    '''compute J_phi plasma using fitting curve
    - psi : poloidal magnetic flux / 2pi
    - psi_a : psi on magnetic axis
    - psi_b : psi on plasma boundary(X-point)
    - r : radius from axis to current position
    - r0 : radius form axis to center
    - lamda : coeff(update)
    - beta_0 : coeff(update)
    - m : coeff(fixed)
    - n : coeff(fixed)
    '''
    
    psi_s = (psi - psi_a) / (psi_b - psi_a + EPS)
    
    if lamda is None:
        return (beta_0 * (r/r0) + (1-beta_0) * (r0/r)) * (1-psi_s ** m) ** n
    else:
        return lamda * (beta_0 * (r/r0) + (1-beta_0) * (r0/r)) * (1-psi_s ** m) ** n

def compute_derivative_matrix(A_origin, x_min : float, x_max : float, y_min : float, y_max : float, axis = 0):
    '''compute derivative of matrix dA/dB while B : x or y axis
    - (option) axis : 0 or 1, if 0 then dA/dR, while 1 then dA/dZ
    - x_min, x_max : range of radius
    - y_min, y_max : range of height
    - error order : 2
    '''
    dev_A = np.zeros_like(A_origin)
    
    n_mesh_x = A_origin.shape[0]
    n_mesh_y = A_origin.shape[1]
    
    dx = (x_max - x_min) / (n_mesh_x - 1)
    dy = (y_max - y_min) / (n_mesh_y - 1)
    
    if axis == 0:
        for idx_x in range(0, n_mesh_x):
            
            if idx_x == 0:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            elif idx_x == n_mesh_x - 1:
                dev = A_origin[idx_x, :] - A_origin[idx_x - 1, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            else:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x - 1, :]
                dev /= (2 * dx)
                dev_A[idx_x, :] = dev
        
    else:
        for idx_y in range(0, n_mesh_y):
            if idx_y == 0:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y]
                dev /= dy
                dev_A[:,idx_y] = dev
            elif idx_y == n_mesh_y - 1:
                dev = A_origin[:, idx_y] - A_origin[:, idx_y - 1]
                dev /= dy
                dev_A[:,idx_y] = dev
            else:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y - 1]
                dev /= (2 * dy)
                dev_A[:,idx_y] = dev
            
    return dev_A

def compute_2nd_derivative_matrix(A_origin, x_min : float, x_max : float, y_min : float, y_max : float, axis = 0):
    '''compute 2nd derivative of matrix d^2A/dx^2, d^2A/dy^2 or d^2A/dxdy
    - (option) axis : 0 or 1, if 0 then dA/dR, while 1 then dA/dZ
    - x_min, x_max : range of radius
    - y_min, y_max : range of height
    - error order : 2
    '''
    dev_A = np.zeros_like(A_origin)
    
    n_mesh_x = A_origin.shape[0]
    n_mesh_y = A_origin.shape[1]
    
    dx = (x_max - x_min) / (n_mesh_x - 1)
    dy = (y_max - y_min) / (n_mesh_y - 1)
    
    if axis == 0:
        for idx_x in range(0, n_mesh_x):
            
            if idx_x == 0:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            elif idx_x == n_mesh_x - 1:
                dev = A_origin[idx_x, :] - A_origin[idx_x - 1, :]
                dev /= dx
                dev_A[idx_x, :] = dev
                
            else:
                dev = A_origin[idx_x + 1, :] - A_origin[idx_x - 1, :]
                dev /= (2 * dx)
                dev_A[idx_x, :] = dev
        
    else:
        for idx_y in range(0, n_mesh_y):
            if idx_y == 0:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y]
                dev /= dy
                dev_A[:,idx_y] = dev
            elif idx_y == n_mesh_y - 1:
                dev = A_origin[:, idx_y] - A_origin[:, idx_y - 1]
                dev /= dy
                dev_A[:,idx_y] = dev
            else:
                dev = A_origin[:, idx_y + 1] - A_origin[:, idx_y - 1]
                dev /= (2 * dy)
                dev_A[:,idx_y] = dev
            
    return dev_A