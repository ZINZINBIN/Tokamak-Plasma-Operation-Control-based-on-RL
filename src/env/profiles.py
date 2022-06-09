import numpy as np
import abc
from abc import ABCMeta, abstractmethod
from scipy.integrate import romb, quad
from src.env.utils.physical_constant import MU
from src.env.critical import find_critical, core_mask


class Profile(metaclass = ABCMeta):
    '''
    RHS : F(psi) and P(psi) -> Fp and pressure with respect to psi_norm
    methods : pprime(psi_norm), ffprime(psi_norm)
    '''
    @abstractmethod
    def ffprime(self, pn):
        pass

    @abstractmethod
    def pprime(self, pn):
        pass

    @abstractmethod
    def Jtor(self, R, Z, psi, psi_bndry = None):
        pass
    
    def pressure(self, psi_norm, out = None):
        if not hasattr(psi_norm, "shape"):
            val, _ = quad(self.pprime, psi_norm, 1.0)
            return val * (self.psi_axis - self.psi_bndry)

        if out is None:
            out = np.zeros(psi_norm.shape)
        
        pvals = np.reshape(psi_norm, -1)
        ovals = np.reshape(out, -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays have different length")
        
        for i in range(len(pvals)):
            val, _ = quad(self.pprime, pvals[i], 1.0)
            val *= self.psi_axis - self.psi_bndry
            ovals[i] = val

        return np.reshape(ovals, psi_norm.shape)

    def fpol(self, psi_norm, out = None):
        if not hasattr(psi_norm, "__len__"):
            val, _ = quad(self.ffprime, psi_norm, 1.0)
            val *= self.psi_axis - self.psi_bndry

            return np.sqrt(2.0 * val + self.fvac() ** 2)
        
        psi_norm = np.array(psi_norm)

        if out is None:
            out = np.zeros(psi_norm.shape)
        
        pvals = np.reshape(psi_norm, -1)
        ovals = np.reshape(out, -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays have different length")
        
        for i in range(len(pvals)):
            val, _ = quad(self.ffprime, pvals[i], 1.0)
            val *= self.psi_axis - self.psi_bndry

            ovals[i] = np.sqrt(2.0 * val + self.fvac() ** 2)

        return np.reshape(ovals, psi_norm.shape)
    

class ConstraintBetapIp(Profile):
    '''
    Constrain poloidal Beta Bp and plasma current Ip
        betap : poloidal beta
        Ip : plasma current [Amps]
        fvac : vacuum f = R * Bt
        Raxis : R sued in p' and ff' components
    '''

    def __init__(self, betap, Ip, fvac, alpha_m = 1.0, alpha_n = 2.0, Raxis = 1.0):
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")
        
        self.betap = betap
        self.Ip = Ip
        self.fvac = fvac
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.Raxis = Raxis
    
    def Jtor(self, R, Z, psi, psi_bndry = None):
        '''Calculate toroidal plasma current
        Jtor = L * (Beta0 * R / Raxis + (1-Beta0) * Raxis / R) * jtorshape
        jtorshape : shape function of Jtor
        L and Beta0 : parameters which are set by constraints
        '''

        opt, xpt = find_critical(R,Z,psi)

        if not opt:
            raise ValueError("No O-point found")
        
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = core_mask(R,Z,psi,opt,xpt,psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = core_mask(R,Z,psi,opt,xpt)
        else:
            psi_bndry = psi[0,0]
            mask = None