from src.utils.backend import xp, zeros, to_numpy

# ---- Create array
x = zeros((2, 3))

print("Array type:", type(x))
print("Array:\n", x)

# ---- Convert to numpy
x_np = to_numpy(x)
print("Converted type:", type(x_np))