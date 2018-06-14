import numpy as np
from utils import *
from numpy import pi, log2, real, exp, sqrt, abs, sum, mean, diag, power, min, max, argmin, argmax
from numpy.linalg import det, cholesky as chol
import matplotlib.pyplot as plt

np.random.seed(233)

# Basic configurations
Nt = 64     # No. transmit antennas
U = 4       # No. RF chains & users
Nch = 5     # No. channel paths
Nrf = U     # No. RF chains
Nr_set = np.array([1, 2, 4, 8])[0:U]     # Users' receive antennas
# Nr_set = np.array([1, 1, 1, 1])[0:U]     # Users' receive antennas
Nr_total = sum(Nr_set)

D_prc_sw = True

# Generate channel matrix
dec_fac = 0.8   # Path gain decaying factor
aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(dec_fac, U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

snr_db_rng = np.linspace(-20, 10, 20)

# Step 1: Design analog precoder
A_prc_mat = np.mat(np.zeros(shape=[Nt, U], dtype='complex'))
for u_id in range(U):
    temp = np.mat(a_resp_vec(aod_set[u_id, 0], Nt))/sqrt(Nt)
    A_prc_mat[:, u_id] = temp.T

G_mat = H_mat * A_prc_mat # Equivalent channel

# Step 2: Design digital precoder
if D_prc_sw and max(Nr_set) == 1:
    print('Not: Digital precoder is ON.')
    D_prc_mat = G_mat.H * (G_mat * G_mat.H).I
    D_prc_mat = D_prc_mat / sqrt(fnorm2(D_prc_mat)) * sqrt(U)
elif D_prc_sw and max(Nr_set) > 1:
    print('Warning: Digital precoder is OFF due to multiple RAs.')
    D_prc_mat = np.eye(U)
else:
    print('Warning: Digital precoder is OFF due to off-switch.')
    D_prc_mat = np.eye(U)

T_mat = G_mat * D_prc_mat

# Step 3: Calculate per-user data rate
bl_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
bl_ami_pfm = np.zeros_like(snr_db_rng)
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)
    for u_id in range(U):
        Nu = Nr_set[u_id]
        
        Hu = T_mat[sum(Nr_set[0:u_id]):sum(Nr_set[0:u_id])+Nu, :]
        sig_pow = fnorm2(Hu[:, u_id])
        N1 = N0 + (fnorm2(Hu) - sig_pow) / Nu
        bl_mi_pfm[u_id, snr_id] = log2(1 + sig_pow/N1)
        bl_ami_pfm[snr_id] += bl_mi_pfm[u_id, snr_id] / U

for u_id in range(U):
    plt.plot(snr_db_rng, bl_mi_pfm[u_id, :], '-', label='BL-user {0}'.format(u_id))

plt.plot(snr_db_rng, bl_ami_pfm, 'ko-', label='bl-mean')
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Mutual Information (bits/s/Hz)')
plt.grid()
plt.show()
    





