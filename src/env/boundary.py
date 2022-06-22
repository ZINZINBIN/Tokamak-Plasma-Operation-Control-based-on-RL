from scipy.integrate import romb
import numpy as np
from src.env.utils.GreenFunction import GreenFunctionScaled, GreenFunction

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

    R = eq.R
    Z = eq.Z

    nx, ny = psi.shape

    bndry_indices = np.concatenate(
        [
            [(x,0) for x in range(nx)],
            [(x,ny-1) for x in range(nx)],
            [(0,y) for y in range(ny)],
            [(nx-1,y) for y in range(ny)]
        ]
    )

    for x,y in bndry_indices:

        greenfunc = GreenFunction(R,Z,R[x,y],Z[x,y])

        greenfunc[x,y] = 0

        psi[x,y] = romb(romb(greenfunc * Jtor)) * dR * dZ