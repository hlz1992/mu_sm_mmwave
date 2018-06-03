import numpy as np
import matplotlib.pyplot as plt
from utils import *
from numpy import pi, log2, real, exp, sqrt, abs
from numpy.linalg import det, cholesky as chol


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
Mu_set = np.array([1, 2, 2, 4])[0:U]    # Users' hyper-RF chain number
wu_set = dict()                         # User weights set
for u_id in range(U):
    wu_set[u_id] = 1









