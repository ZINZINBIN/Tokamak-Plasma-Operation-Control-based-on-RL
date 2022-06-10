import numpy as np
import matplotlib.pyplot as plt
from src.env.critical import find_critical, core_mask
from src.env.environment import Coil
from src.env.equilibrium import Equilibrium

def plotCoils(coils, axis = None):
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
    
    return axis

def plotConstraints(control, axis = None, show = True):
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)

    # Locations of the X-points
    for r, z in control.xpoints:
        axis.plot(r, z, "bx")

    if control.xpoints:
        axis.plot([], [], "bx", label="X-point constraints")

    # Isoflux surfaces
    for r1, z1, r2, z2 in control.isoflux:
        axis.plot([r1, r2], [z1, z2], ":b^")

    if control.isoflux:
        axis.plot([], [], ":b^", label="Isoflux constraints")

    if show:
        plt.legend()
        plt.show()

    return 

def plotEquilibrium(eq : Equilibrium, axis = None, show : bool= True, oxpoints : bool= True, wall : bool= True):
    R = eq.R
    Z = eq.Z

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
    
    levels = np.linspace(np.amin(eq.psi), np.amax(eq.psi), 100)

    axis.contour(R,Z,eq.psi, levels = levels)
    axis.set_aspect("equal")
    axis.set_xlabel("Major radius [m]")
    axis.set_ylabel("Height [m]")

    if oxpoints:
        opt, xpt = find_critical(R,Z,eq.psi, True)

        for r,z,_ in xpt:
            axis.plot(r,z,'ro')
        
        for r,z,_ in opt:
            axis.plot(r,z,'go')
        
        if xpt:
            psi_bndry = xpt[0][2]
            axis.contour(eq.R, eq.Z, eq.psi, levels=[psi_bndry], colors="r")

            axis.plot([], [], "ro", label="X-points")
            axis.plot([], [], "r", label="Separatrix")

        if opt:
            axis.plot([], [], "go", label="O-points")

    if wall and eq.device.wall and len(eq.device.wall.R):
        
        axis.plot(
            list(eq.device.wall.R) + [eq.device.wall.R[0]],
            list(eq.device.wall.Z) + [eq.device.wall.Z[0]],
            "k",
        )

    if show:
        plt.legend()
        plt.show()

    return axis