import numpy as np
from scipy import special
import torch, math
from torch.autograd import Function
import torch.nn.functional as F
from typing import Union, List, Dict
import math

'''
class GreenFunction(Function):
    @staticmethod
    def forward(ctx, R0 : torch.Tensor, Z0:torch.Tensor, R:torch.Tensor, Z:torch.Tensor):
        R0, Z0, R, Z = R0.detach(), Z0.detach(), R.detach(), Z.detach()
        k = np.sqrt(4 * R0.numpy() * R.numpy() / ((R.numpy() + R0.numpy()) ** 2 + (Z.numpy() - Z0.numpy()) ** 2))
        ellipK = special.ellipk(k)
        ellipE = special.ellipe(k)
        g = np.sqrt(R0.numpy() * R.numpy()) / 2 / math.pi / k * ((2-k**2) * ellipK - 2 * ellipE)
        return g
'''

def gradient(u : torch.Tensor, x : torch.Tensor):
    u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), retain_graph = True, create_graph=True)[0]
    return u_x

def compute_Jphi(
    psi_s : torch.Tensor, 
    R : torch.Tensor,
    Rc : Union[torch.Tensor, float],
    alpha_m : torch.Tensor, 
    alpha_n : torch.Tensor, 
    beta_m : torch.Tensor, 
    beta_n : torch.Tensor, 
    lamda : torch.Tensor, 
    beta : torch.Tensor
    ):
    
    Jp = torch.pow((1 - torch.pow(psi_s, alpha_m.to(psi_s.device))), alpha_n.to(psi_s.device)) * R / Rc
    Jf = torch.pow((1 - torch.pow(psi_s, beta_m.to(psi_s.device))), beta_n.to(psi_s.device)) * Rc / R
    
    Jphi = Jp * beta + Jf * (1-beta)
    Jphi *= lamda
    return Jphi

def compute_pprime(
    psi_s : np.array, 
    R : np.array,
    Rc : Union[np.array, float],
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : float, 
    beta : float
    ):
    pprime = np.power((1 - np.power(psi_s, alpha_m)), alpha_n) / Rc
    pprime *= beta * lamda
    return pprime

def compute_ffprime(
    psi_s : np.array, 
    R : np.array,
    Rc : Union[np.array, float],
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : float, 
    beta : float
    ):
    ffprime = np.power((1 - np.power(psi_s, beta_m)), beta_n) * Rc
    ffprime *= (1-beta) * lamda
    return ffprime

def compute_Jphi_1D(
    psi_s : np.array, 
    R : np.array,
    Rc : Union[np.array, float],
    alpha_m : float, 
    alpha_n : float, 
    beta_m : float, 
    beta_n : float, 
    lamda : float, 
    beta : float
    ):
    
    Jp = np.power((1 - np.power(psi_s, alpha_m)), alpha_n) * R / Rc
    Jf = np.power((1 - np.power(psi_s, beta_m)), beta_n) * Rc / R
    
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
def compute_grad_shafranov_loss(psi : torch.Tensor, R : torch.Tensor, Z : torch.Tensor, Jphi : torch.Tensor):
    loss = eliptic_operator(psi, R, Z) + R * Jphi * 4 * math.pi * 10 ** (-7)
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