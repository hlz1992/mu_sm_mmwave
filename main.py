import numpy as np
from utils import *
from numpy import pi, log2, real, exp, sqrt, abs, sum, diag, power
from numpy.linalg import det, cholesky as chol

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

np.random.seed(13)

# Basic configurations
Nt = 64     # No. transmit antennas
U = 4       # No. RF chains
Nch = 5     # No. channel paths
Nrf = U     # No. RF chains
Nr_set = np.array([1, 2, 4, 8])[0:U]     # Users' receive antennas

# Generate channel matrix
aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

# User-interference-cancellation precoder design
mu_set = np.array([1, 2, 2, 4])[0:U]    # Users' hyper-RF chain number
wu_set = dict()                         # User weights set
for u_id in range(U):
    mu = mu_set[u_id]
    wu_set[u_id] = np.ones(shape=[mu]) / sqrt(U)

# Construct analog precoder
M_total = sum(mu_set)
A_mat = np.mat(np.zeros(shape=[Nt, M_total], dtype='complex'))
for u_id in range(U):
    mu = mu_set[u_id]

    A_u = np.mat(a_resp_mat(aod_set[u_id, 0:mu], Nt) / sqrt(Nt))
    A_u = A_u * np.mat(diag(wu_set[u_id]))
    A_mat[:, sum(mu_set[0:u_id]):sum(mu_set[0:u_id])+mu] = A_u

T_mat = power(abs(Q_mat.H * A_mat), 2)
# Z = T_mat/T_mat.max()
# print(Z)
# plt.matshow(Z)
# plt.show()

# Analog-domain channel
HA_mat = H_mat * A_mat

