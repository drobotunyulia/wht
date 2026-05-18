import numpy as np
from classes_16 import HADAMARD_16_CLASSES
from classes_24 import HADAMARD_24_CLASSES

def check_hadamard(H):
    n = H.shape[0]
    return np.allclose(H @ H.T, n * np.eye(n))

for name, H in HADAMARD_24_CLASSES.items():
    print(name, check_hadamard(H))