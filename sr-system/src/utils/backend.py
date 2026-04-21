# ---- Backend abstraction layer

import numpy as np

# ---- Try importing CuPy (GPU)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# ---- Default backend (will connect to config later)
BACKEND = "numpy"

# ---- Select backend
if BACKEND == "cupy" and CUPY_AVAILABLE:
    xp = cp
elif BACKEND == "numpy":
    xp = np
else:
    xp = np


# ---- Move array to configured device
def to_device(array):
    if BACKEND == "cupy" and CUPY_AVAILABLE:
        return cp.asarray(array)
    return np.asarray(array)


# ---- Convert to numpy (CPU)
def to_numpy(array):
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


# ---- Create zeros
def zeros(shape, dtype=np.float32):
    return xp.zeros(shape, dtype=dtype)


# ---- Create ones
def ones(shape, dtype=np.float32):
    return xp.ones(shape, dtype=dtype)


# ---- Random array (useful later)
def randn(*shape):
    return xp.random.randn(*shape)