import numpy as np
import abc
from abc import ABCMeta, abstractmethod
from scipy.integrate import romb, quad
from src.env.utils.physical_constant import MU, pi
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

    @abstractmethod
    def fvac(self):
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
        self._fvac = fvac
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

        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]

        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        jtor_shape = (1.0 - np.clip(psi_norm, 0, 1.0) ** self.alpha_m) ** self.alpha_n

        if mask is not None:
            jtor_shape *= mask

        # pshape : integral of jtor_shape        
        def pshape(psi_n):
            shape_integral, _ = quad(
                lambda x : (1.0 - x ** self.alpha_m) ** self.alpha_n, psi_n, 1.0
            )
            shape_integral *= (psi_bndry - psi_axis)
            return shape_integral

        # P(psi) = -(L * beta0 / Raxis) * pshape(psi_norm)

        nx, ny = psi_norm.shape
        pfunc = np.zeros((nx,ny))
        
        for i in range(1,nx -1):
            for j in range(1,ny-1):
                if((psi_norm[i,j]>=0) and (psi_norm[i,j] < 1.0)):
                    pfunc[i,j] = pshape(psi_norm)

        if mask is not None:
            pfunc *= mask

        # Integrate over plasma
        # betap = (8pi/mu0) * int(p)dRdZ / Ip^2
        #       = - (8pi/mu0) * (L*Beta0/Raxis) * intp / Ip^2
        intp = romb(romb(pfunc)) * dR * dZ

        LBeta0 = -self.betap * (MU / (8.0 * pi)) * self.Raxis * self.Ip ** 2 / intp

        IR = romb(romb(jtor_shape * R / self.Raxis)) * dR * dZ
        I_R = romb(romb(jtor_shape * self.Raxis / R)) * dR * dZ

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtor_shape

        self.L = L
        self.Beta0 = Beta0
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    def pprime(self, pn):
        shape = (1.0 - np.clip(pn, 0, 1) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape
    
    def ffprime(self, pn):
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return MU * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac

class ConstraintPaxisIp(Profile):
    '''
    Constrain Pressure at magnetic axis and plasma current Ip
        paxis : pressure at magnetic axis
        Ip : plasma current [Amps]
        fvac : vacuum f = R * Bt
        Raxis : R sued in p' and ff' components
    '''
    def __init__(self, paxis, Ip, fvac, alpha_m = 1.0, alpha_n = 2.0, Raxis = 1.0):
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_n < 0:
            raise ValueError("alpha_n must be positive")
        
        self.paxis = paxis
        self.Ip = Ip
        self._fvac = fvac
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

        dR = R[1,0] - R[0,0]
        dZ = Z[0,1] - Z[0,0]

        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        jtor_shape = (1.0 - np.clip(psi_norm, 0, 1.0) ** self.alpha_m) ** self.alpha_n

        if mask is not None:
            jtor_shape *= mask

        shapeintegral, _ = quad(
            lambda x : (1.0 - x ** self.alpha_m) ** self.alpha_n,
            0,
            1.0
        )

        shapeintegral *= psi_bndry - psi_axis

        # P(psi) = -(L * beta0 / Raxis) * pshape(psi_norm)
        IR = romb(romb(jtor_shape * R / self.Raxis)) * dR * dZ
        I_R = romb(romb(jtor_shape * self.Raxis / R)) * dR * dZ

        LBeta0 = -self.paxis * self.Raxis / shapeintegral        

        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtor_shape

        self.L = L
        self.Beta0 = Beta0
        self.psi_bndry = psi_bndry
        self.psi_axis = psi_axis

        return Jtor

    def pprime(self, pn):
        shape = (1.0 - np.clip(pn, 0, 1) ** self.alpha_m) ** self.alpha_n
        return self.L * self.Beta0 / self.Raxis * shape
    
    def ffprime(self, pn):
        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        return MU * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        return self._fvac