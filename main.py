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
Nr_total = sum(Nr_set)

# Generate channel matrix
aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

# Construct analog & digital precoder
# Step 1: Determine hyper-RF-chain number for each user
Mu_set = np.array([1, 2, 2, 4])[0:U]    # Users' hyper-RF chain number

# Step 2: Determine per-hyper-RF-chain weight for each user
wu_set = dict()                         # User weights set
for u_id in range(U):
    Mu = Mu_set[u_id]
    wu_set[u_id] = np.ones(shape=[Mu]) / sqrt(U)

# Step 3: Calculate corresponding hybrid precoders
M_total = sum(Mu_set)
A_prc_set = dict()
D_prc_set = dict()
P_prc_set = dict()
P_mat = np.mat(np.zeros(shape=[Nt, M_total], dtype='complex'))
for u_id in range(U):
    Mu = Mu_set[u_id]

    A_prc_set[u_id] = np.mat(a_resp_mat(aod_set[u_id, 0:Mu], Nt)) / sqrt(Nt)
    D_prc_set[u_id] = np.mat(diag(wu_set[u_id]))
    P_prc_set[u_id] = A_prc_set[u_id] * D_prc_set[u_id]

    P_mat[:, sum(Mu_set[0:u_id]):sum(Mu_set[0:u_id])+Mu] = P_prc_set[u_id]

# Equivalent channel
G_mat = H_mat * P_mat
Gu_set = dict()
for u_id in range(U):
    Nu = Nr_set[u_id]
    Mu = Mu_set[u_id]

    Gu_set[u_id] = G_mat[sum(Nr_set[0:u_id]):sum(Nr_set[0:u_id])+Nu, sum(Mu_set[0:u_id]):sum(Mu_set[0:u_id])+Mu]

# print([a.shape for a in Gu_set.values()])

# Calculate accurate per-user mutual information


