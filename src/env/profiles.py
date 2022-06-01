import numpy as np
from scipy.integrate import romb, quad
from src.env.utils.physical_constant import MU

class Profile:
    def __init__(self, Jtor, pprime, ffprime, pressure, fpol):
        self.Jtor = Jtor
        self.pprime = pprime
        self.ffprime = ffprime
        self.pressure = pressure
        self.fpol = fpol
