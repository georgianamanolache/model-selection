import numpy as np

class Alpine1d(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, x):
        x = -x
        rval = - (x * np.sin(x + (self.shift * np.pi / 12)) + 0.1 * x)
        if isinstance(x, np.ndarray):
            if len(x) == 1:
                return rval[0]
            else:
                return rval.flatten()
        else:
            return rval