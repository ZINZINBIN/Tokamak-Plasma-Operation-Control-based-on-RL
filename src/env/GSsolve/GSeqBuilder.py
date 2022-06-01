import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix, eye
from typing import Optional, Tuple, List
from src.env.utils.GreenFunction import GreenBr, GreenBz, GreenFunctionScaled, GreenFunction
from src.env.utils.physical_constant import MU

class GSElliptic:
    '''Calculate the Grad-Shafranov eliptic operator
    '''
    def __init__(self, Rmin : float, Rmax : float, Zmin : float, Zmax : float):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
    
    def __call__(self, psi:np.ndarray):
        nx = psi.shape[0]
        ny = psi.shape[1]

        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        A = np.zeros(nx, ny)

        dR2_inv = 1.0 / dR ** 2
        dZ2_inv = 1.0 / dZ ** 2
        
        for idx_x in range(1, nx -1):

            R = self.Rmin + dR * idx_x

            for idx_y in range(1, ny-1):
                A[idx_x, idx_y] = (-2) * (dR2_inv + dZ2_inv) * psi[idx_x, idx_y] 
                + dR2_inv * (psi[idx_x + 1, idx_y] + psi[idx_x - 1, idx_y])
                + dZ2_inv * (psi[idx_x, idx_y + 1] + psi[idx_x, idx_y - 1])
                + (-1) * (psi[idx_x + 1, idx_y] - psi[idx_x - 1, idx_y]) / (2 * R * dR)
        
        return A


class GSsparse:
    '''Calculate the sparse matrices for the GS operator(not using psi)
    '''
    def __init__(self, Rmin : float, Rmax : float, Zmin : float, Zmax : float):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

    def __call__(self, nx, ny):
        
        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        N = nx * ny

        A = eye(N, format = "lil")

        dR2_inv = 1.0 / dR ** 2
        dZ2_inv = 1.0 / dZ ** 2
        
        for idx_x in range(1, nx -1):

            R = self.Rmin + dR * idx_x

            for idx_y in range(1, ny-1):
                
                idx = idx_x * ny + idx_y
                A[idx, idx - 1] = dZ2_inv
                A[idx, idx - ny] = dR2_inv + 1.0 / (2.0 * R * dR)
                A[idx, idx] = -2.0 * (dZ2_inv + dR2_inv)

                A[idx, idx + ny] = dR2_inv - 1.0 / (2.0 * R * dR)

                A[idx, idx + 1] = dZ2_inv
        
        return A.tocsr()

class GSsparse4thOrder:
    '''Calculate the sparse matrices for the GS operator with 4th order FDM(not using psi)
    '''
    def __init__(self, Rmin : float, Rmax : float, Zmin : float, Zmax : float):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

        self.centred_1st = [
            (-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)
        ]

        self.offset_1st = [
            (-1, -3.0 / 12),
            (0, -10.0 / 12),
            (1, 18.0 / 12),
            (2, -6.0 / 12),
            (3, 1.0 / 12),
        ]

        self.centred_2nd = [
            (-2, -1.0 / 12),
            (-1, 16.0 / 12),
            (0, -30.0 / 12),
            (1, 16.0 / 12),
            (2, -1.0 / 12),
        ]

        self.offset_2nd = [
            (-1, 10.0 / 12),
            (0, -15.0 / 12),
            (1, -4.0 / 12),
            (2, 14.0 / 12),
            (3, -6.0 / 12),
            (4, 1.0 / 12),
        ]

    def __call__(self, nx, ny):
        
        dR = (self.Rmax - self.Rmin) / (nx - 1)
        dZ = (self.Zmax - self.Zmin) / (ny - 1)

        N = nx * ny
        A = lil_matrix((N,N))

        dR2_inv = 1.0 / dR ** 2
        dZ2_inv = 1.0 / dZ ** 2
        
        for idx_x in range(1, nx -1):

            R = self.Rmin + dR * idx_x

            for idx_y in range(1, ny-1):
                
                idx = idx_x * ny + idx_y

                if idx_y == 1:
                    for offset, weight in self.offset_2nd:
                        A[idx, idx + offset] += weight * dZ2_inv

                elif idx_y == ny - 2:
                    for offset, weight in self.offset_2nd:
                        A[idx, idx - offset] += weight * dZ2_inv
                else:
                    for offset, weight in self.centred_2nd:
                        A[idx, idx + offset] += weight * dZ2_inv

                
                if idx_x == 1:
                    for offset, weight in self.offset_2nd:
                        A[idx, idx + offset * ny] += weight * dR2_inv

                    for offset, weight in self.offset_1st:
                        A[idx, idx + offset * ny] -= weight / (R * dR)
                
                elif idx_x == nx - 2:
                    for offset, weight in self.offset_2nd:
                        A[idx, idx - offset * ny] += weight * dR2_inv

                    for offset, weight in self.offset_1st:
                        A[idx, idx - offset * ny] += weight / (R * dR)
                
                else:
                    for offset, weight in self.centred_2nd:
                        A[idx, idx + offset * ny] += weight * dR2_inv

                    for offset, weight in self.centred_1st:
                        A[idx, idx + offset * ny] -= weight / (R * dR)

        for idx_x in range(nx):
            for idx_y in [0, ny - 1]:
                idx = idx_x * ny + idx_y
                A[idx,idx] = 1.0

        for idx_x in [0, nx - 1]:
            for idx_y in range(ny):
                idx = idx_x * ny + idx_y
                A[idx,idx] = 1.0

        return A.tocsr()
      