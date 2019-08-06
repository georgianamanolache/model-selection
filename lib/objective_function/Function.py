import numpy as np

class Function(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, x):
        # Shift is hardcoded 5 to replicate the FB BoTorch
        rval = (-1) * 0.1 * (x - 1) * (np.sin(x + 5) + 0.1) + 5 + self.shift - 1
        if isinstance(x, np.ndarray):
            if len(x) == 1:
                return rval[0]
            else:
                return rval.flatten()
        else:
            return rval