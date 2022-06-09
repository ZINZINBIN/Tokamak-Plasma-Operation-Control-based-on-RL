from cv2 import multiply
import numpy as np
import numbers
from typing import List, Optional, Tuple, Union
from src.env.utils.GreenFunction import GreenFunctionScaled, GreenBr, GreenBz
from src.env.utils.physical_constant import MU, pi

class AreaCurrentLimit:
    def __init__(self, current_density : float = 3.5e9):
        self.current_density = current_density

    def __call__(self, coil):
        return abs(coil.current * coil.turns) / self.current_density

class Coil:
    def __init__(self, R :float, Z : float, current : float = 0.0, turns : int = 1, control : bool = True, area : AreaCurrentLimit = AreaCurrentLimit()):
        self.R = R
        self.Z = Z
        self.mu = MU
        self.current = current
        self.turns = turns
        self.control = control
        self.area = area

    def __repr__(self):
        return "Coil(R = {0}, Z = {1}, current = {2:.1f}, turns = {3}, control : {4}".format(self.R, self.Z, self.current, self.turns, self.control)
    
    def __eq__(self, other):
        return(
            self.R == other.R and
            self.Z == other.Z and
            self.current == other.current and
            self.turns == other.turns and
            self.control == other.control
        )

    def __ne__(self, other):
        return not self == other
    
    @property
    def area(self):
        if isinstance(self._area, numbers.Number):
            assert self._area > 0
            return self._area
        
        area = self._area(self)
        assert area >0
        return area
    
    @area.setter
    def area(self, area):
        self._area = area

    def _controlPsi(self, R, Z):
        ''' return G(R,Z) without scaled factor(dimensionless)
        '''
        return GreenFunctionScaled(self.R, self.Z, R, Z) * self.turns

    def _controlBr(self, R, Z):
        Br = GreenBr(self.R, self.Z, R, Z, False) * self.turns
        return Br
    
    def _controlBz(self, R, Z):
        Bz = GreenBz(self.R, self.Z, R, Z, False) * self.turns
        return Bz
    
    def Br(self, R, Z):
        return self._controlBr(R,Z) * self.current * self.mu
    
    def Bz(self, R, Z):
        return self._controlBz(R,Z) * self.current * self.mu

    def psi(self, R, Z):
        return self._controlPsi(R,Z) * self.current * self.mu 
    
    def createPsiGreens(self, R, Z):
        return self._controlPsi(R,Z) * self.mu
    
    def calcPsiFromGreens(self, pgreen):
        return self.current * pgreen * self.mu

    def calcForces(self, equilibrium):
        '''calculate forces on the coils(N), returns an array of two element : [Fr, Fz]
        * Force on coils occurs due to its own current : Lorentz self‐forces on curved current loops
        * 논문 참조 : Physics of Plasmas 1, 3425 (1998); https://doi.org/10.1063/1.870491 (David A. Garren and James Chen)
        '''

        current = self.current
        total_current = current * self.turns

        Br = equilibrium.Br(self.R, self.Z)
        Bz = equilibrium.Bz(self.R, self.Z)

        minor_radius = np.sqrt(self.area, pi)

        self_inductance = 0.5

        force = self.mu * total_current ** 2 / (4.0 * pi * self.R) * (np.log(8.0 * self.R / minor_radius) - 1 + self_inductance / 2.0)

        len_coils = 2 * pi * self.R

        return np.array(
            [
                (total_current * Bz + force) * len_coils, - total_current * Br * len_coils
            ]
        )

    def plot(self, axis=None, show=False):
        """
        Plot the coil location, using axis if given
        The area of the coil is used to set the radius
        """
        minor_radius = np.sqrt(self.area / np.pi)

        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        circle = plt.Circle((self.R, self.Z), minor_radius, color="b")
        axis.add_artist(circle)
        return axis
    

class Solenoid(object):
    def __init__(self, Rs : float, Zsmin : float, Zsmax : float, Ns : int, current : float= 0, control :bool = True):
        self.Rs = Rs
        self.Zsmin = Zsmin
        self.Zsmax = Zsmax
        self.Ns = Ns
        self.current = current
        self.control = control
        self.mu = MU

    def _controlPsi(self, R, Z):
        result = 0

        for Zs in np.linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += GreenFunctionScaled(self.Rs,Zs,R,Z)

        return result

    def createPsiGreens(self, R, Z):
        return self._controlPsi(R,Z) * self.mu
    
    def calcPsiFromGreens(self, pgreen):
        return self.current * pgreen * self.mu
    
    def _controlBr(self, R, Z):
        result = 0

        for Zs in np.linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += GreenBr(self.Rs, Zs, R, Z, scaled = False)

        return result

    def _controlBz(self, R, Z):
        result = 0
        for Zs in np.linspace(self.Zsmin, self.Zsmax, self.Ns):
            result += GreenBz(self.Rs, Zs, R, Z, scaled=False)

        return result

    def Br(self, R, Z):
        return self._controlBr(R,Z) * self.mu * self.current

    def Bz(self, R, Z):
        return self._controlBz(R,Z) * self.mu * self.current

    def psi(self, R, Z):
        return self._controlPsi(R,Z) * self.mu * self.current

    def __repr__(self):
        return "Solenoid(Rs={0}, Zsmin={1}, Zsmax={2}, current={3}, Ns={4}, control={5})".format(
            self.Rs, self.Zsmin, self.Zsmax, self.current, self.Ns, self.control
        )

    def __eq__(self, other):
        return (
            self.Rs == other.Rs
            and self.Zsmin == other.Zsmin
            and self.Zsmax == other.Zsmax
            and self.Ns == other.Ns
            and self.current == other.current
            and self.control == other.control
        )

    def __ne__(self, other):
        return not self == other

    def plot(self, axis=None, show=False):
        return axis

class Wall(object):
    def __init__(self, R_min : float, R_max : float, Z_min : float, Z_max : float):
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max

        self.R = [R_min, R_max]
        self.Z = [Z_min, Z_max]
    
    def __repr__(self):
        return "Wall(R_min = {R_min}, R_max = {R_max}, Z_min={Z_min}, Z_max = {Z_max}".format(R_min=self.R_min, R_max = self.R_max, Z_min = self.Z_min, Z_max = self.Z_max)
    
    def __eq__(self, other):
        return np.allclose(self.R, other.R) and np.allclose(self.Z, other.Z)
    
    def __ne__(self, other):
        return not self == other

class Circuit(object):
    def __init__(self, coils : List[Tuple[str, Coil, float]], current : float = 0, control : bool = True):
        self.coils = coils
        self.current = current
        self.control = control

    def _controlPsi(self, R, Z):
        result = 0

        for label, coil, multiplier in self.coils:
            result += multiplier * coil._controlPsi(R,Z)

        return result

    def psi(self, R, Z):
        psi_val = 0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            psi_val += coil.psi(R,Z)
        return psi_val
    
    def createPsiGreens(self, R, Z):
        pgreen = {}
        for label, coil, multiplier in self.coils:
            pgreen[label] = coil.createPsiGreens(R,Z)
        return pgreen
    
    def calcPsiFromGreens(self, pgreen):
        psi_val = 0
        for label, coil, multiplier in self.coils:
            coil.current = self.current * multiplier
            psi_val += coil.calcPsiFromGreens(pgreen[label])
        return psi_val
    
    def _controlBr(self, R, Z):
        result = 0

        for label, coil, multiplier in self.coils:
            result += multiplier * coil._controlBr(R,Z)

        return result

    def _controlBz(self, R, Z):
        result = 0

        for label, coil, multiplier in self.coils:
            result += multiplier * coil._controlBz(R,Z)

        return result

    def calcForces(self, equilibrium):
        forces = {}
        for label, coil, multiplier in self.coils:
            forces[label] = coil.calcForces(equilibrium)
        return forces

    def __repr__(self):
        result = "Circuit(["
        coils = [
            '("{0}", {1}, {2})'.format(label, coil, multiplier)
            for label, coil, multiplier in self.coils
        ]
        result += ", ".join(coils)
        return result + "], current={0}, control={1})".format(
            self.current, self.control
        )

    def __eq__(self, other):
        return (
            self.coils == other.coils
            and self.current == other.current
            and self.control == other.control
        )

    def __ne__(self, other):
        return not self == other

    def plot(self, axis=None, show=False):
        for label, coil, multiplier in self.coils:
            axis = coil.plot(axis=axis, show=False)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return axis


class Device(object):
    def __init__(self, coils : List[Tuple[str, Union[Coil, Circuit, Solenoid]]], wall : Wall):
        super(Device, self).__init__()
        self.coils = coils
        self.wall = wall

    def __repr__(self):
        return "Device(coils = {coils}, wall = {wall}".format(coils = self.coils, wall = self.wall)
    
    def __eq__(self, other):
        return sorted(self.coils) == sorted(other.coils) and self.wall == other.wall
    
    def __ne__(self, other):
        return not self == other
    
    def __getitem__(self, name : str):
        for label, coil in self.coils:
            if label == name:
                return coil
        raise KeyError("Device does not contail coil with label : {0}".format(name))

    def psi(self, R, Z):
        psi_coils = 0
        for label, coil in self.coils:
            psi_coils += coil.psi(R,Z)
        return psi_coils

    def createPsiGreens(self, R, Z):
        pgreen = {}
        for label, coil in self.coils:
            pgreen[label] = coil.createPsiGreens(R,Z)
        return pgreen

    def calcPsiFromGreens(self, pgreen):
        psi_coils = 0
        for label, coil in self.coils:
            psi_coils += coil.calcPsiFromGreens(pgreen[label])
        return psi_coils
    
    def Br(self, R, Z):
        Br = 0
        for label, coil in self.coils:
            Br += coil.Br(R,Z)
        return Br
    
    def Bz(self, R, Z):
        Bz = 0
        for label, coil in self.coils:
            Bz += coil.Bz(R,Z)
        return Bz

    def _controlBr(self, R, Z):
        return [coil._controlBr(R,Z) for label, coil in self.coils if coil.control]
    
    def _controlBz(self, R, Z):
        return [coil._controlBz(R,Z) for label, coil in self.coils if coil.control]
    
    def _controlPsi(self, R, Z):
        return [coil._controlPsi(R,Z) for label, coil in self.coils if coil.control]
    
    def controlAdjust(self, current_change):
        control_coils = [coil for label, coil in self.coils]

        for coil, dI in zip(control_coils, current_change):
            coil.current += dI.item()
    
    def controlCurrent(self):
        return [coil.current for label, coil in self.coils if coil.control]
    
    def setControlCurrent(self, currents):
        controlCoils = [coil for label, coil in self.coils if coil.control]
        for coil, current in zip(controlCoils, currents):
            coil.current = current

    def printCurrents(self):
        print("==========================")
        for label, coil in self.coils:
            print(label + " : " + str(coil))
        print("==========================")

    def calcForces(self, equilibrium = None):
        if equilibrium is None:
            equilibrium = self

        forces = {}
        for label, coil in self.coils:
            forces[label] = coil.calcForces(equilibrium)
        return forces

    def calcCurrents(self):
        currents = {}

        for label, coil in self.coils:
            currents[label] = coil.current
        return currents
    
    def plot(self, axis=None, show=True):
        for label, coil in self.coils:
            axis = coil.plot(axis=axis, show=False)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return axis