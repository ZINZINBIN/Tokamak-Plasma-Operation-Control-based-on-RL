import numpy as np
import random

class Buffer(object):
    def __init__(self, env, n_steps : int, max_size: int = 100000, raw_pixels : bool = False):