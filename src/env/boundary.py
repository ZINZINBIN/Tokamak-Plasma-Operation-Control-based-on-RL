from scipy.integrate import romb
import numpy as np

def FixedBoundary(eq, Jtor : np.ndarray, psi : np.ndarray):
    psi[0,:] = 0
    psi[:,0] = 0
    psi[-1, :] = 0
    psi[:, -1] = 0

def FreeBoundary(eq, Jtor : np.ndarray, psi : np.ndarray):
    '''Apply a free boundary using von Hagenow's method
    Inputs : eq(object), Jtor(2D_array), psi(2D_array)
    returns : None
    '''
    dR = eq.dR
    dZ = eq.dZ

    nx, ny = psi.shape

    rhs = eq.R * Jtor

    rhs[0,:] = 0
    rhs[:,0] = 0
    rhs[-1,:] = 0
    rhs[:,-1] = 0
