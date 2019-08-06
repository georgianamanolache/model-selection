import numpy as np

class Alpine2d(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, x):
        rval = np.sqrt(np.abs(x)) * np.sin(x + self.shift - 1) + self.shift - 1
        if isinstance(x, np.ndarray):
            if len(x) == 1:
                return rval[0]
            else:
                return rval.flatten()
        else:
            return rval