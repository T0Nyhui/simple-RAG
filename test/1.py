import sys
import numpy
import scipy

print(f"Python Executable: {sys.executable}")
print(f"Numpy Version: {numpy.__version__} | Path: {numpy.__file__}")
print(f"Scipy Version: {scipy.__version__} | Path: {scipy.__file__}")

try:
    from scipy.special import _multiufuncs
    print("Direct import success")
except Exception as e:
    print(f"Error detail: {e}")