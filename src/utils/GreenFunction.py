import numpy as np
from scipy import special
from numpy import clip
import math

pi = math.pi
MU = 4 * pi * 10 ** (-7) # permuability for vaccum
EPS = 1e-6
dv_ratio = 1e-2

def GreenFunction(R0, Z0, R, Z, mu = MU):
    '''GreenFunction for toroidal geometry
    Calculate polidal magnetic flux at (R,Z) from a unit current at (R0,Z0)
    '''

    k = np.sqrt(4 * R0 * R / ((R + R0) ** 2 + (Z - Z0) ** 2))
    k = clip(k, 1e-10, 1.0, -1e-10)

    ellipK = special.ellipk(k)
    ellipE = special.ellipe(k)
    g = mu * np.sqrt(R*R0) / 2 / pi / k * ((2-k**2) * ellipK - 2 * ellipE)
    return g

def GreenFunctionScaled(R0, Z0, R, Z):
    '''GreenFunction for toroidal geometry(dimensionless)
    Calculate polidal magnetic flux at (R,Z) from a unit current at (R0,Z0)
    '''

    k = np.sqrt(4 * R0 * R / ((R + R0) ** 2 + (Z - Z0) ** 2))
    k = clip(k, 1e-10, 1.0, -1e-10)

    ellipK = special.ellipk(k)
    ellipE = special.ellipe(k)
    g = np.sqrt(R*R0) / 2 / pi / k * ((2-k**2) * ellipK - 2 * ellipE)

    return g

def GreenBz(R0, Z0, R,Z, mu, Ic, dr : float = 0.001, scaled : bool = True):

    Gi = GreenFunctionScaled(R0,Z0,R - 0.5 * dr,Z)
    Gf = GreenFunctionScaled(R0,Z0,R + 0.5 * dr,Z)
    dGdr = (Gf - Gi) / dr
    Bz = 1 / R * dGdr
    scaled_factor = Ic * mu

    if scaled:
        return Bz * scaled_factor
    else:
        return Bz, scaled_factor

def GreenBr(R0, Z0, R, Z, mu, Ic, dz : float = 0.001, scaled : bool = True):

    Gi = GreenFunction(R0, Z0, R, Z - 0.5 * dz, mu)
    Gf = GreenFunction(R0, Z0, R, Z + 0.5 * dz, mu)
    dGdz = (Gf - Gi) / dz

    scaled_factor = Ic * mu
    Br = (-1) * dGdz / R

    if scaled:
        return Br * scaled_factor
    else:
        return Br, scaled_factor

def GreenFunctionMatrix(R, Z, R_min, Z_min, R_max, Z_max, Nr, Nz, mu = MU):
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    G = np.zeros((Nr, Nz))
    
    for idx_r in range(0,Nr):
        for idx_z in range(0,Nz):
            R0 = R_min + idx_r * dR
            Z0 = Z_min + idx_z * dZ
            G[idx_r, idx_z] = GreenFunction(R0, Z0, R, Z, mu)      
    return G
 
def GreenFunctionMatrixScaled(R, Z, R_min, Z_min, R_max, Z_max, Nr, Nz):
    dR = (R_max - R_min) / (Nr - 1)
    dZ = (Z_max - Z_min) / (Nz - 1)
    G = np.zeros((Nr, Nz))
    
    for idx_r in range(0,Nr):
        for idx_z in range(0,Nz):
            R0 = R_min + idx_r * dR
            Z0 = Z_min + idx_z * dZ
            G[idx_r, idx_z] = GreenFunctionScaled(R0, Z0, R, Z)      
    return G     