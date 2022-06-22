import numpy as np
from numpy.linalg import inv
from scipy import optimize
from typing import List, Tuple, Optional, Union
from src.env.critical import find_critical, find_psisurface, find_safety, find_separatrix
from src.env.equilibrium import Equilibrium
from src.env.visualize import plotConstraints

class Constrain(object):
    '''
    Adjust coil currents using constraint condition

    (ex) ConstrainSystem = Constrain(xpoints = [(1.0,1.1), (1.0, 0.4)])
    >>> In this case, system will attempt to create xpoints at (R,Z) = (1.0,1.1) and (1.0,0.4)

    argument
    - xpoints : a list of x-point (R,Z)
    - gamma : a scalar, minimise the magnitude of the coil currents
    - isoflux : a list of tuple (R1,Z1, R2,Z2)
    - psival : a list of (R,Z,psi)
    '''
    def __init__(self, xpoints : List = [], gamma : float = 1e-12, isoflux : List = [], psival : List = []):
        self.xpoints = xpoints
        self.gamma = gamma
        self.isoflux = isoflux
        self.psival = psival

    def __call__(self, eq:Equilibrium):
        device = eq.getDevice()

        constraint_matrix = []
        constraint_rhs = []
        
        # In x-points, magnetic field Br and Bz should be zero
        for xpt in self.xpoints:
            Br = eq.Br(xpt[0],xpt[1])

            constraint_rhs.append(-Br) # add current to cancel out the Br field
            constraint_matrix.append(device._controlBr(xpt[0], xpt[1]))

            Bz = eq.Bz(xpt[0], xpt[1])

            constraint_rhs.append(-Bz) # also add current to cancel out the Bz field
            constraint_matrix.append(device._controlBz(xpt[0], xpt[1]))

        # constrain point to have same flux
        for r1,z1,r2,z2 in self.isoflux:
            p1 = eq.psiRZ(r1,z1)
            p2 = eq.psiRZ(r2,z2)
            constraint_rhs.append(p2-p1)

            # coil response
            c1 = device._controlPsi(r1,z1)
            c2 = device._controlPsi(r2,z2)

            c = [c1val - c2val for c1val, c2val in zip(c1,c2)]

            constraint_matrix.append(c)
        
        # constrain the value of psi
        for r,z,psi in self.psival:
            p1 = eq.psiRZ(r,z)
            constraint_rhs.append(psi - p1)

            # coil response
            c = device._controlPsi(r,z)
            constraint_matrix.append(c)

        if not constraint_rhs:
            raise ValueError("No contraints given")

        A = np.array(constraint_matrix)
        b = np.array(constraint_rhs).reshape(-1)

        # solve constraint condition by Tikhonov regularization

        # number of control
        ncontrols = A.shape[1]

        current_change = np.dot(
            inv(np.dot(A.T,A) + self.gamma ** 2 * np.eye(ncontrols)),
            np.dot(A.T, b)
        )

        device.controlAdjust(current_change)

        eq._constraints = self
    
    def plot(self, axis = None, show : bool = True):
        return plotConstraints(self, axis = axis, show = show)

class ConstrainPsi2D(object):
    def __init__(self, target_psi:np.ndarray, weights : Optional[np.ndarray] = None):
        if weights is None:
            weights = np.full(shape = target_psi.shape, fill_value=1.0)

        self.target_psi = target_psi - np.average(target_psi, weights = weights)
        self.weights = weights
    
    def __call__(self, eq : Equilibrium):
        device = eq.getDevice()
        start_currents = device.controlCurrent()

        end_currents, _ = optimize.leastsq(
            self.psi_difference,
            start_currents,
            args = (eq,)
        )

        device.setControlCurrent(end_currents)
        eq._constraints = self

    def psi_difference(self, currents, eq:Equilibrium):
        eq.getDevice().setControlCurrent(currents)
        psi = eq.psi()

        psi_av = np.average(psi, weights=self.weights)
        psi_diff = (psi - psi_av - self.target_psi) * self.weights

        return psi_diff.ravel() # flatten array

