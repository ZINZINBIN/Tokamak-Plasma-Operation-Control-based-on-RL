from matplotlib.pyplot import plasma
from sklearn import multiclass
import numpy as np
import logging
from scipy import interpolate
from scipy.integrate import romb
from src.env.GSsolve.GSeqBuilder import GSsparse, GSsparse4thOrder
from src.env.critical import find_critical, core_mask, find_safety, find_separatrix, find_psisurface
from src.env.environment import Device, EmptyTokamak
from src.env.boundary import FreeBoundary, FixedBoundary
from src.env.utils.multigrid import createVcycle
from src.env.utils.physical_constant import pi, MU, K
from src.env.visualize import plotEquilibrium
from src.env.profiles import Profile, ConstraintBetapIp, ConstraintPaxisIp

logger = logging.getLogger()

class Equilibrium:
    def __init__(
        self,
        device : Device = EmptyTokamak(),
        boundary = FreeBoundary,
        psi = None,
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
        self._apply_boundary = boundary
        self._constraints = None
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
        self._updatePlasmaPsi(psi)
        
        # setting for generator : GS matrix class
        if order == 2:
            gs_matrix = GSsparse(Rmin, Rmax, Zmin, Zmax)
        elif order == 4:
            gs_matrix = GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
        else:
            order = 2
            gs_matrix = GSsparse(Rmin, Rmax, Zmin, Zmax)
        
        self.order = order

        self._solver = createVcycle(
            nx, ny, gs_matrix, n_levels = 1, n_cycle = 1, n_iter = 2, direct = True
        )

    # solver : setter and getter
    def setSolverVcycle(self, n_levels = 1, n_cycle = 1, n_iter = 1, direct = True):
        generator = GSsparse(self.Rmin, self.Rmax, self.Zmin, self.Zmax)
        nx, ny = self.R.shape

        self._solver = createVcycle(
            nx,ny,generator,n_levels,n_cycle,n_iter,direct
        )

    def setSolver(self, solver):
        self._solver = solver
    
    # call solver so that returns the solution psi with A*psi = rhs
    def callSolver(self, psi, rhs):
        return self._solver(psi, rhs)

    # get device
    def getDevice(self):
        return self.device
    
    # calculate and get physical variables : Br, Bz, Ip, Btor, psi
    def plasmaCurrent(self):
        return self.current
    
    def poloidalBeta(self):
        dR = self.dR
        dZ = self.dZ

        psi_norm = (self.psi() - self.psi_axis) / (self.psi_bndry - self.psi_axis)

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
        psi_norm = (self.psiRZ(R,Z) - self.psi_axis) / (self.psi_bndry - self.psi_axis)

        fpol = self.fpol(psi_norm)

        if self.mask is not None:
            mask = self.mask_func(R,Z,grid = False)
            fpol = fpol * mask + (1.0 - mask) * self.fvac()

        return fpol / R
    
    def psi(self):
        return self.plasma_psi + self.device.calcPsiFromGreens(self._pgreen)

    def psiRZ(self, R, Z):
        return self.psi_func(R,Z,grid = False) + self.device.psi(R,Z)

    def fpol(self, psi_norm):
        return self._profiles.fpol(psi_norm)
    
    def fvac(self):
        return self._profiles.fvac()
    
    def q(self, psi_norm = None, n_psi = 128):
        if psi_norm is None:
            psi_norm = np.linspace(1.0 / (n_psi + 1), 1.0, n_psi, endpoint = False)
            return psi_norm, find_safety(self, psi_norm = psi_norm)
        
        result = find_safety(self, psi_norm = psi_norm)

        if len(result) == 1:
            return np.asscalar(result)
        
        return result

    def pressure(self, psi_norm):
        return self._profiles.pprime(psi_norm)

    def separatrix(self, n_theta : int = 128):
        sep = find_separatrix(self, psi = self.psi(), n_theta = n_theta)
        return np.array(sep[:,0:2])

    def _updatePlasmaPsi(self, plasma_psi : np.ndarray):

        self.plasma_psi = plasma_psi

        # update spline interpolation
        self.psi_func = interpolate.RectBivariateSpline(
            self.R[:,0], self.Z[0,:], plasma_psi
        )

        psi = self.psi()

        opt, xpt = find_critical(self.R, self.Z, plasma_psi)

        if opt:
            self.psi_axis = opt[0][2]

            if xpt:
                self.psi_bndry = xpt[0][2]
                self.mask = core_mask(self.R, self.Z, psi, opt, xpt, self.psi_bndry)
                self.mask_func = interpolate.RectBivariateSpline(
                    self.R[:,0],
                    self.Z[0,:],
                    self.mask
                )
            elif self._apply_boundary == FixedBoundary:
                # No x-point but fixed boundary
                self.psi_bndry = psi[0,0]
                self.mask = None
            else:
                self.psi_bndry = psi[0,0] # None
                self.mask = None

    def plot(self, axis = None, show : bool= True, oxpoints : bool = True, wall : bool = True):
        return plotEquilibrium(self, axis = axis, show = show, oxpoints=oxpoints, wall = wall)

    def solve(self, profiles : Profile, Jtor = None, psi = None, psi_bndry = None):
        
        # update profiles
        self._profiles = profiles

        # calculate Jtor 
        if Jtor is None:
            if psi is None:
                psi = self.psi()
            
            Jtor = profiles.Jtor(self.R,self.Z,psi, psi_bndry=psi_bndry)

        # apply boundary condition
        self._apply_boundary(self,Jtor,self.plasma_psi)

        # obtain RHS of GS equation
        rhs = -MU*self.R*Jtor

        # Copy boundary conditions
        rhs[0, :] = self.plasma_psi[0, :]
        rhs[:, 0] = self.plasma_psi[:, 0]
        rhs[-1, :] = self.plasma_psi[-1, :]
        rhs[:, -1] = self.plasma_psi[:, -1]

        # solve GS equation
        plasma_psi = self._solver(self.plasma_psi, rhs)
        
        # update plasma psi
        self._updatePlasmaPsi(plasma_psi)

        # update plasma current 
        dR = self.R[1,0] - self.R[0,0]
        dZ = self.Z[0,1] - self.Z[0,0]

        self.current = romb(romb(Jtor)) * dR * dZ

    # force calculation
    def getForces(self):
        return self.device.calcForces(self)
    
    def printForces(self):

        print("Forces on coils")

        def print_forces(forces:dict, prefix=""):
            for label, force in forces.items():
                if isinstance(force, dict):
                    print(prefix + label + " (circuit)")
                    print_forces(force, prefix=prefix + "  ")
                else:
                    print(
                        prefix
                        + label
                        + " : R = {0:.2f} kN , Z = {1:.2f} kN".format(
                            force[0] * 1e-3, force[1] * 1e-3
                        )
                    )

        print_forces(self.getForces())
        
def solve(
    eq : Equilibrium, 
    profiles : Profile, 
    constrain = None,
    rtol : float = 1e-3,
    atol : float = 1e-10,
    blend : float = 0.0,
    show : bool = False,
    axis = None,
    pause : float = 0.0001,
    psi_bndry = None,
    maxits : int = 64,
    convergenceInfo : bool = False
    ):

    if constrain is not None:
        constrain(eq)
    
    psi = eq.psi()

    if show:
        import matplotlib.pyplot as plt
        if pause > 0 and axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
    
    psi_maxchange_iterations, psi_relchange_iterations = [], []

    is_converged = False

    for iter in range(maxits):
        if show:
            if pause < 0:
                fig = plt.figure()
                axis = fig.add_subplot(111)
            else:
                axis.clear()
            
            plotEquilibrium(eq, axis, show = False)

            if pause < 0:
                plt.show()
            else:
                axis.figure.canvas.draw()
                plt.pause(pause)

        psi_last = psi.copy()

        eq.solve(profiles, psi = psi, psi_bndry=psi_bndry)

        psi = eq.psi()

        psi_change = psi_last - psi
        psi_maxchange = np.amax(abs(psi_change))
        psi_relchange = psi_maxchange / (np.amax(psi) - np.amin(psi))

        psi_maxchange_iterations.append(psi_maxchange)
        psi_relchange_iterations.append(psi_relchange)

        # Check if the relative change in psi is small enough
        if (psi_maxchange < atol) or (psi_relchange < rtol):
            is_converged = True
            break

        # Adjust the coil currents
        if constrain is not None:
            constrain(eq)

        psi = (1.0 - blend) * eq.psi() + blend * psi_last
    
    if is_converged:
        print("Picard iterations converge...!")
    else:
        print("Picard iterations failed to converge(iteration over)")
    
    if convergenceInfo:
        return np.array(psi_maxchange_iterations), np.array(psi_relchange_iterations)

# test for equilibrium
def test_fixed_boundary_psi():

    profiles = ConstraintPaxisIp(
        1e3, 1e5, 1.0  # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
    )  # fvac = R*Bt

    eq = Equilibrium(
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=FixedBoundary
    )

    # Nonlinear solve
    solve(eq, profiles)

    psi = eq.psi()
    assert psi[0, 0] == 0.0  # Boundary is fixed
    assert psi[32, 32] != 0.0  # Solution is not all zero

    assert eq.psi_bndry == 0.0
    assert eq.poloidalBeta() > 0.0


def test_fixed_boundary_psi():
    # This is adapted from example 5

    profiles = ConstraintPaxisIp(
        1e3, 1e5, 1.0  # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
    )  # fvac = R*Bt

    eq = Equilibrium(
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=FixedBoundary,
    )

    # Nonlinear solve
    solve(eq, profiles, maxits = 64)

    psi = eq.psi()
    assert psi[0, 0] == 0.0  # Boundary is fixed
    assert psi[32, 32] != 0.0  # Solution is not all zero

    assert eq.psi_bndry == 0.0
    assert eq.poloidalBeta() > 0.0

def test_setSolverVcycle():
    eq = Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65)
    oldsolver = eq._solver
    eq.setSolverVcycle(n_levels = 2, n_cycle = 1, n_iter = 5)
    assert eq._solver != oldsolver