import numpy as np


class Griewank(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, x):
        rval = x ** 2 / 4000 - np.cos(x / 3 + (self.shift * np.pi / 12))
        if isinstance(x, np.ndarray):
            if len(x) == 1:
                return rval[0]
            else:
                return rval.flatten()
        else:
            return rval