import numpy as np
from utils import *
from numpy import pi
from numpy.linalg import det, cholesky as chol
from numpy import pi
import matplotlib.pyplot as plt

Nr = 8
N = int(1e3)
y = np.zeros(N)
for k in range(N):
    temp = [np.exp(-1j*pi*n*k/N) for n in range(Nr)]
    y[k] = np.sum(temp)

k_rng = np.array([k/N for k in range(N)])
y2 = np.exp(-1j*pi*(Nr-1)*k_rng/2) * np.sin(pi*Nr*k_rng/2) / np.sin(pi*k_rng/2)

plt.plot(y, 'b-')
plt.plot(y2, 'r--')
plt.show()