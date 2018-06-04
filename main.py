import numpy as np
from utils import *
from numpy import pi, log2, real, exp, sqrt, abs, sum, mean, diag, power
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
T_mat = np.zeros(shape=[U, U])
for u_id in range(U):
    Nu = Nr_set[u_id]
    Mu = Mu_set[u_id]

    Gu_set[u_id] = G_mat[sum(Nr_set[0:u_id]):sum(Nr_set[0:u_id])+Nu, sum(Mu_set[0:u_id]):sum(Mu_set[0:u_id])+Mu]

    for u2_id in range(U):
        T_mat[u_id, u2_id] = fnorm2(G_mat[sum(Nr_set[0:u_id]):sum(Nr_set[0:u_id])+Nu, sum(Mu_set[0:u2_id])]) / Nu

# Calculate accurate per-user mutual information
# Part-I: Single-antenna MI
snr_db_rng = np.linspace(-20, 20, 20)
sa_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
sa_ami_pfm = np.zeros_like(snr_db_rng)
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)

    for u_id in range(U):
        Nu = Nr_set[u_id]
        h_sa = Gu_set[u_id][:, 0]
        sa_mi_pfm[u_id, snr_id] = log2(real(det(np.eye(Nu) + h_sa*h_sa.H / N0)))
        sa_ami_pfm[snr_id] += sa_mi_pfm[u_id, snr_id] / U

# for u_id in range(U):
#     plt.plot(snr_db_rng, sa_mi_pfm[u_id, :], '-', label='user {0}'.format(u_id))
plt.plot(snr_db_rng, sa_ami_pfm, 'k-', label='sa-mean')

# Part-II: Spatial-modulation MI
sm_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
sm_ami_pfm = np.zeros_like(snr_db_rng)
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)

    for u_id in range(U):
        Nu = Nr_set[u_id]
        Mu = Mu_set[u_id]
        Gu = Gu_set[u_id]
        
        sym_dm_mi = mean([log2(real(det(np.eye(Nu) + Gu[:, col_id]*Gu[:, col_id].H / N0))) for col_id in range(Mu)])

        # spt_dm_mi = log2(Mu)
        spt_dm_mi = 0

        sm_mi_pfm[u_id, snr_id] = sym_dm_mi + spt_dm_mi
        sm_ami_pfm[snr_id] += sm_mi_pfm[u_id, snr_id] / U

plt.plot(snr_db_rng, sm_ami_pfm, 'k--', label='sm-mean')

# Picture
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Mutual Information (bits/s/Hz)')
plt.grid()
plt.show()


