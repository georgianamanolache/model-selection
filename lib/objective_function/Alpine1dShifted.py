import numpy as np

class Alpine1dShifted(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, x):
        rval = (-1) * 0.1 * (x - 1) * (np.sin(x + self.shift - 1) + 0.1) + self.shift - 1
        if isinstance(x, np.ndarray):
            if len(x) == 1:
                return rval[0]
            else:
                return rval.flatten()
        else:
            return rval