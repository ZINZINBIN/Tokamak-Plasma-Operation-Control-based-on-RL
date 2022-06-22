import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from src.env.boundary import FreeBoundary
from src.env.control import Constrain
from src.env.equilibrium import Equilibrium, solve
from src.env.environment import Device, EmptyTokamak
from src.env.profiles import ConstraintPaxisIp


if __name__ == "__main__":
    device = EmptyTokamak()
    eq = Equilibrium(
        device = device,
        boundary = FreeBoundary,
        Rmin = 0.1,
        Rmax = 2.0,
        Zmin = -1.0,
        Zmax = 1.0,
        nx = 65,
        ny = 65,
        order = 4
    )

    profiles = ConstraintPaxisIp(
        paxis = 1e3,
        Ip = 2e5,
        fvac = 2.0
    )

    xpoints = [
        (1.1,-0.6),
        (1.1, 0.8)
    ]

    isoflux = [
        (1.1, -0.6, 1.1, 0.6)
    ]

    constrain = Constrain(
        xpoints=xpoints,
        isoflux = isoflux,
        psival = []
    )

    solve(
        eq = eq,
        profiles = profiles,
        constrain = constrain,
        convergenceInfo=True
    )

    print("solve free boundary condition of GS equation - done!")
    
    print("\n=========== Info ===========\n")
    print("plasma current : {:.3f}(A)".format(eq.plasmaCurrent()))
    print("Plasma pressure on axis: {:.3f}(Pa)".format(eq.pressure(0.0)))
    print("Poloidal beta: {:.3f}".format(eq.poloidalBeta()))

    # print the whole information about current
    device.printCurrents()

    # Forces on the coils
    eq.printForces()

    print("\nSafety factor:\n\tpsi \t q")

    for psi in [0.01, 0.9, 0.95]:
        print("\t{:.2f}\t{:.2f}".format(psi, eq.q(psi)))

    axis = eq.plot(show=False)
    eq.device.plot(axis=axis, show=False)
    constrain.plot(axis=axis, show=True)

    plt.plot(*eq.q())
    plt.xlabel(r"Normalised $\psi$")
    plt.ylabel("Safety factor")
    plt.savefig("./results/GS-Solver/q-freeBoundary.png")
    plt.show()