''' Code for ploting the flux boundary under the condition of soloviev equilibrium
    In this case, we assume that z = (1/r) * sqrt(2 * R^2 * kappa * psi_b /Bc - (1/4) * kappa^2 * (r^2-R^2)^2) => soloviev equilibrium
    We can obtain plasma boundary if we have information which contains as below
    - tri-top and tri-bot
    - kappa
    - R-major and a-minor
    - B-centric
    - psi for boundary
    Then, we use finite difference method to solve Grad-shafranov equation as a fixed-boundary condition.
    The details of the process are referred from https://iopscience.iop.org/article/10.1088/1742-6596/1159/1/012017/pdf
'''
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, List, Union, Literal

def compute_zb(r : Union[np.array, float], R0 : float, kappa : float, q0 : float, Bc : float, psi_b : float):
    eps = 1e-3
    if type(r) == float:
        zb = 1 / (r + eps) * math.sqrt(2 * R0 ** 2 * kappa * q0 / Bc * psi_b - 0.25 * kappa ** 2 * (r ** 2 - R0 **2) **2)
    else:
        zb = np.sqrt(2 * R0 ** 2 * kappa * q0 / Bc * psi_b - 0.25 * kappa ** 2 * (r ** 2 - R0 **2) **2)
        zb /= r + eps
        
    return zb

def compute_ri(z : Union[np.array, float],  R0 : float,kappa : float, q0 : float, Bc : float, psi_b : float):
    r = np.sqrt(R0**2 - 2 / kappa ** 2 * z ** 2 + np.sqrt(4 * z ** 4 / kappa ** 4 - z ** 2 * R0 ** 2 + 2 * R0 ** 2 * kappa * q0 * psi_b / Bc))
    return r

class SolovievEquilibrium:
    def __init__(
        self, 
        n : int,
        ds : float,
        R0 : float,
        a : float,
        q0 : float,
        kappa : float,
        tritop : float,
        tribot : float,
        Bc : float,
        psi_b : float,
        ):
        
        # argument
        self.n = n
        self.R0 = R0
        self.a = a
        self.q0 = q0
        self.kappa = kappa
        self.tritop = tritop
        self.tribot = tribot
        self.Bc = Bc
        
        self.Rmin = R0 - a
        self.Rmax = R0 + a
        
        self.psi_b = psi_b
        
        self.ds = ds
        
        
    def compute_upper_boundary_point(self):
        
        # regime 1 : (Rmin, R0)
        r1 = np.linspace(self.Rmin, self.R0 - self.tritop * self.a, self.n)
        z1 = compute_zb(r1, self.R0 - self.tritop * self.a, self.kappa, self.q0, self.Bc, self.psi_b)
        
        # regime 2 : (R0, Rmax)
        
    def compute_lower_boundary_point(self):
        pass
    
    def scale_transform(self):
        pass
    
    def scale_inverse_transform(self):
        pass
    
    def solve_GS(self):
        # scaling
        
        # compute the upper boundary point
        
        # compute the lower boundary point
        
        # generate mesh grid inside the boundary
        
        # solve nonlinear equation
        # process 1. generate FDM
        
        # process 2. choose iterative solve or direct solve
        
        # process 3. rescale
        
        pass
    
    def plot(self, *arg):
        pass