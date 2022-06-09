from sklearn import multiclass
import numpy as np
import logging
from scipy import interpolate
from scipy.integrate import romb
from src.env.GSsolve.GSeqBuilder import GSsparse, GSsparse4thOrder
from src.env.environment import Device
from src.env.boundary import FreeBoundary, FixedBoundary
from src.env.utils.multigrid import createVcycle
from src.env.utils.physical_constant import pi, MU, K

logger = logging.getLogger()

class Equilibrium:
    def __init__(
        self,
        device : Device,
        boundary,
        psi,
        mask = None, # x-point or limiter
        current = 0,
        Rmin:float = 0.1,
        Rmax:float=2.0,
        Zmin:float=-1.0,
        Zmax:float=1.0,
        nx : int = 64,
        ny : int = 64,
        order : int = 4
    ):
        '''
        Equilibrium
        member variables : equilibrium state including plasma and coil currents

        Argument
        - Rmin, Rmax, Zmin, Zmax : Range of major radius and height
        - nx, ny : Resolution in R and Z, must me 2n + 1
        - boundary : the boundary condition, FreeBoundary or FixedBoundary
        - psi : magnetic flux with 2D array
        - current : plasma current
        - order : GS matrix order
        '''
        self.device = device
        self.boundary = boundary
        self.mask = mask

        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax

        self.R_1D = np.linspace(Rmin, Rmax, nx)
        self.Z_1D = np.linspace(Zmin, Zmax, ny)

        self.dR = (Rmax - Rmin) / nx
        self.dZ = (Zmax - Zmin) / ny
        
        self.R, self.Z = np.meshgrid(self.R_1D, self.Z_1D, indexing = "ij")

        if psi is None:
            xx, yy = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny), indexing = "ij")
            psi = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.4 ** 2)

            psi[0, :] = 0.0
            psi[:, 0] = 0.0
            psi[-1, :] = 0.0
            psi[:, -1] = 0.0

        self._pgreen = device.createPsiGreens(self.R, self.Z)
        
        self.current = current
        self.plasma_psi = None

        # update psi with init value
        self.updatePlasmaPsi(psi)
        
        # setting for generator : GS matrix class
        if order == 2:
            gs_matrix = GSsparse(Rmin, Rmax, Zmin, Zmax)
        elif order == 4:
            gs_matrix = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
        else:
            order = 2
            gs_matrix = GSsparse(Rmin, Rmax, Zmin, Zmax)
        
        self.order = order

        self.solver = createVcycle(
            nx, ny, gs_matrix, n_levels = 1, n_cycle = 1, n_iter = 2, direct = True
        )

    # solver : setter and getter
    def setSolverVcycle(self, n_levels = 1, n_cycle = 1, n_iter = 1, direct = True):
        generator = GSsparse(self.Rmin, self.Rmax, self.Zmin, self.Zmax)
        nx, ny = self.R.shape

        self.solver = createVcycle(
            nx,ny,generator,n_levels,n_cycle,n_iter,direct
        )

    def setSolver(self, solver):
        self.solver = solver
    
    # call solver so that returns the solution psi with A*psi = rhs
    def callSolver(self, psi, rhs):
        return self.solver(psi, rhs)

    # get device
    def getDevice(self):
        return self.device
    
    # calculate and get physical variables : Br, Bz, Ip, Btor, psi
    def plasmaCurrent(self):
        return self.current
    
    def poloidalBeta(self):
        dR = self.dR
        dZ = self.dZ

        psi_norm = (self.psi() - self.psi_axis) / (self.psi_boundary - self.psi_axis)

        pressure = self.pressure(psi_norm)

        if self.mask is not None:
            pressure *= self.mask

        # Integrate pressure in 2D 
        return (
            ((8.0 * pi) / MU)
            * romb(romb(pressure))
            * dR
            * dZ
            / (self.plasmaCurrent() ** 2)
        )
        
    # calculate the volume of the plasma
    def plasmaVolume(self):
        dR = self.dR
        dZ = self.dZ

        dV = 2.0 * pi * self.R * dR * dZ

        if self.mask is not None:
            dV *= self.mask
        
        return romb(romb(dV))

    def plasmaBr(self, R, Z):
        return -self.psi_func(R, Z, dy = 1, grid = False) / R
    
    def plasmaBz(self, R, Z):
        return self.psi_func(R,Z, dx = 1, grid = False) / R

    def Br(self, R, Z):
        return self.plasmaBr(R,Z) + self.device.Br(R,Z)
    
    def Bz(self, R, Z):
        return self.plasmaBz(R,Z) + self.device.Bz(R,Z)
    
    def Btor(self, R, Z):
        return None

    def psi(self):
        return self.plasma_psi + self.device.calcPsiFromGreens(self._pgreen)

    def psiRZ(self, R, Z):
        return self.psi_func(R,Z,grid = False) + self.device.psi(R,Z)

    def fpol(self, psi_norm):
        return self.profiles.fpol(psi_norm)
    
    def fvac(self):
        return self.profiles.fvac()

    def pressure(self, psi_norm):
        return self.profiles.pprime(psi_norm)

    def updatePlasmaPsi(self, plasma_psi : np.ndarray):

        self.plasma_psi = plasma_psi

        # update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(
            self.R[:,0], self.Z[0,:], plasma_psi
        )

        psi = self.psi()


        return None
    







