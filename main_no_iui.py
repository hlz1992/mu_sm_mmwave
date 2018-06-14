import numpy as np
from utils import *
from numpy import pi, log2, real, exp, sqrt, abs, sum, mean, diag, power
from numpy.linalg import det, cholesky as chol
import matplotlib.pyplot as plt

np.random.seed(13)

# Basic configurations
Nt = 64     # No. transmit antennas
U = 4       # No. RF chains & users
Nch = 5     # No. channel paths
Nrf = U     # No. RF chains
Nr_set = np.array([1, 2, 4, 8])[0:U]     # Users' receive antennas
Nr_total = sum(Nr_set)

# Generate channel matrix
dec_fac = 0.8   # Path gain decaying factor
aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(dec_fac, U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

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

    A_prc_set[u_id] = np.mat(a_resp_mat(aod_set[u_id, 0:Mu], Nt))/sqrt(Nt)
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

# Calculate accurate per-user mutual information
# Part-I: Single-antenna MI
snr_db_rng = np.linspace(-20, 20, 20)
sa_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
sa_ami_pfm = np.zeros_like(snr_db_rng)
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)
    for u_id in range(U):
        sa_mi_pfm[u_id, snr_id] = sa_mut_inf(Gu_set[u_id][:, 0], N0)
        sa_ami_pfm[snr_id] += sa_mi_pfm[u_id, snr_id] / U

for u_id in range(U):
    plt.plot(snr_db_rng, sa_mi_pfm[u_id, :], '-', label='SA-user {0}'.format(u_id))
plt.plot(snr_db_rng, sa_ami_pfm, 'ko-', label='sa-mean')

# Part-II: Spatial-modulation MI
sm_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
sm_ami_pfm = np.zeros_like(snr_db_rng)

for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)
    for u_id in range(U):
        Gu = Gu_set[u_id]
        sm_mi_pfm[u_id, snr_id], _, _ = sm_mut_inf(Gu, N0)
        sm_ami_pfm[snr_id] += sm_mi_pfm[u_id, snr_id] / U

for u_id in range(U):
    plt.plot(snr_db_rng, sm_mi_pfm[u_id, :], '--', label='SM-user {0}'.format(u_id))
plt.plot(snr_db_rng, sm_ami_pfm, 'kx--', label='sm-mean')

# Picture
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Mutual Information (bits/s/Hz)')
plt.grid()
plt.show()
