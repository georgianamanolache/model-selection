import numpy as np


class Ginuta(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, x):
        rval = 0.6 + np.sin(16 / 15 * x - 1) + np.sin(16 / 15 * x - 1) ** 2 + 1 / 50 * np.sin(
            4 * (16 / 15 * x - 1)) + self.shift-1
        if isinstance(x, np.ndarray):
            if len(x) == 1:
                return rval[0]
            else:
                return rval.flatten()
        else:
            return rval