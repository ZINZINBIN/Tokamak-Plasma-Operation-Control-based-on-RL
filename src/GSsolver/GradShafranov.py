import numpy as np
from scipy import special
import torch, math
from torch.autograd import Function
import torch.nn.functional as F
from typing import Union, List, Dict
from src.GSsolver.KSTAR_setup import limiter_shape
import math

def compute_k(R0 : torch.Tensor, Z0:torch.Tensor, R:torch.Tensor, Z:torch.Tensor):
    k = np.sqrt(4 * R0 * R / ((R + R0) ** 2 + (Z - Z0) ** 2))
    k = np.clip(k, 1e-10, 1 - 1e-10)
    return k

def compute_ellipK_derivative(k, ellipK, ellipE):
    return ellipE / k / (1-k*k) - ellipK / k

def compute_ellipE_derivative(k, ellipK, ellipE):
    return (ellipE - ellipK) / k

def compute_Green_function(R0 : torch.Tensor, Z0:torch.Tensor, R:torch.Tensor, Z:torch.Tensor):
    k = compute_k(R0, Z0, R, Z)
    ellipK = special.ellipk(k)
    ellipE = special.ellipe(k)
    g = 0.5 / math.pi / k * ((2-k**2) * ellipK - 2 * ellipE) * 4 * math.pi * 10 **(-7)
    g *= np.sqrt(R0 * R)
    return g

def gradient(u : torch.Tensor, x : torch.Tensor):
    u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), retain_graph = True, create_graph=True)[0]
    return u_x

def compute_Jphi(
    psi_s : torch.Tensor, 
    R : torch.Tensor,
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : torch.Tensor, 
    beta : torch.Tensor
    ):
    
    Jp = torch.pow((1 - torch.pow(psi_s, alpha_m)), alpha_n) * R
    Jf = torch.pow((1 - torch.pow(psi_s, beta_m)), beta_n) / R
    
    Jphi = Jp * beta + Jf * (1-beta)
    Jphi *= lamda
    return Jphi

def compute_p_psi(
    psi_s : torch.Tensor, 
    R : torch.Tensor,
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : torch.Tensor, 
    beta : torch.Tensor
    ):
    
    def _poly(psi_s, alpha_m, alpha_n):
        result = 0
        for k in range(0, alpha_n):
            result += math.comb(int(alpha_n), k) * psi_s ** (alpha_m * k + 1) / (alpha_m * k + 1) * (-1) ** k           
        return result
    
    p = _poly(psi_s, alpha_m, alpha_n)
    p = p * beta * lamda
    return p

def compute_pprime(
    psi_s : np.array, 
    R : np.array,
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : float, 
    beta : float
    ):
    pprime = np.power((1 - np.power(psi_s, alpha_m)), alpha_n)
    pprime *= beta * lamda
    return pprime

def compute_ffprime(
    psi_s : np.array, 
    R : np.array,
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : float, 
    beta : float
    ):
    ffprime = np.power((1 - np.power(psi_s, beta_m)), beta_n)
    ffprime *= (1-beta) * lamda
    return ffprime

def compute_Jphi_1D(
    psi_s : np.array, 
    R : np.array,
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : float, 
    beta : float
    ):
    
    Jp = np.power((1 - np.power(psi_s, alpha_m)), alpha_n) * R
    Jf = np.power((1 - np.power(psi_s, beta_m)), beta_n) / R
    
    Jphi = Jp * beta + Jf * (1-beta)
    Jphi *= lamda
    return Jphi

def compute_plasma_region(psi_s : torch.Tensor):
    mask = F.relu(1 - psi_s).ge(0).float()
    return mask

# Epliptic operator for GS equation
def eliptic_operator(psi:torch.Tensor, R : torch.Tensor, Z : torch.Tensor):
    psi_r = gradient(psi, R)
    psi_z = gradient(psi, Z)
    psi_z2 = gradient(psi_z, Z)
    psi_r2 = gradient(psi_r, R)
    return psi_r2 - 1 / R * psi_r + psi_z2

# Grad-Shafranov equation as a loss function
def compute_grad_shafranov_loss(psi : torch.Tensor, R : torch.Tensor, Z : torch.Tensor, Jphi : torch.Tensor, Rc : float, psi_s : float):
    loss = eliptic_operator(psi, R, Z) * Rc ** 2 / psi_s + R * Jphi / Rc
    loss = torch.norm(loss)
    return loss

# Determinant : check whether the null point is x-point or axis
def compute_det(psi : torch.Tensor, R : torch.Tensor, Z : torch.Tensor):
    psi_r = gradient(psi, R)
    psi_z = gradient(psi, Z)
    psi_r2 = gradient(psi_r, R)
    psi_z2 = gradient(psi_z, Z)
    det = psi_r2 * psi_z2 - (psi_r * psi_z) ** 2
    return det

def compute_grad2(psi : torch.Tensor, R : torch.Tensor, Z : torch.Tensor):
    psi_r = gradient(psi, R)
    psi_z = gradient(psi, Z)
    grad = psi_r ** 2 + psi_z ** 2
    grad = torch.sqrt(grad)
    return grad

def compute_KSTAR_limiter_mask(RR, ZZ, min_value : float= 5e-2):
    
    def convert_coord_index(RR, ZZ, points_arr):
        indices_arr = []
        for point in points_arr:
            x, y = point

            idx_x, idx_y = 0, 0
            nx,ny = RR.shape
            
            for idx in range(nx-1):
                if RR[0,idx] <= x and RR[0,idx+1] > x:
                    idx_x = idx
                    break
            
            for idx in range(ny-1):
                if ZZ[idx,0] <= y and ZZ[idx+1,0] > y:
                    idx_y = idx
                    break
            
            indices_arr.append([idx_x, idx_y])
        return np.array(indices_arr)
    
    from skimage.draw import polygon
    mask = np.ones_like(RR, dtype = np.float32) * min_value
    contour = convert_coord_index(RR, ZZ, limiter_shape)

    # Create an empty image to store the masked array
    rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
    mask[cc, rr] = 1

    return mask