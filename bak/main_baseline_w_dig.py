#######################################################################
# 
#   @author: Longzhuang He
#   @descrp: Baseline wwave system WITH digital precoding
#
#######################################################################


import numpy as np
from mmwave_utils import *
from numpy import pi, log2, real, exp, sqrt, abs, sum, mean, diag, power, min, max, argmin, argmax
from numpy.linalg import det, cholesky as chol
import matplotlib.pyplot as plt

np.random.seed(233)

# Basic configurations
Nt = 64     # No. transmit antennas
U = 4       # No. RF chains & users
Nch = 5     # No. channel paths
Nrf = U     # No. RF chains
#Nr_set = np.array([1, 2, 4, 8])[0:U]     # Users' receive antennas
Nr_set = np.array([8, 8, 8, 8])[0:U]     # Users' receive antennas
Nr_total = sum(Nr_set)

# Generate channel matrix
dec_fac = 0.7   # Path gain decaying factor
aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(dec_fac, U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

snr_db_rng = np.linspace(-40, 20, 30)    # SNR range (in dB)

# Step 1: Design analog precoder
A_prc_mat = np.mat(np.zeros(shape=[Nt, U], dtype='complex'))
for u_id in range(U):
    temp = np.mat(a_resp_vec(aod_set[u_id, 0], Nt))/sqrt(Nt)
    A_prc_mat[:, u_id] = temp.T

G_mat = H_mat * A_prc_mat # Equivalent channel

# Step 2: Design receiver combining scheme (receiver's digital precoder)
W_mat = np.zeros([Nr_total, U], dtype='complex')
for u_id in range(U):
    Nu = Nr_set[u_id]
    st_idx = sum(Nr_set[0:u_id])
    ed_idx = sum(Nr_set[0:u_id]) + Nu
    W_mat[st_idx:ed_idx, u_id] = a_resp_vec(aoa_set[u_id, 0], Nu)/sqrt(Nu)

W_mat = np.mat(W_mat)
B_mat = W_mat.H * G_mat

# Step 2: Design digital precoder
zf_sw = True
if zf_sw:
    D_prc_mat = B_mat.H * (B_mat * B_mat.H).I
    D_prc_mat = D_prc_mat / sqrt(fnorm2(D_prc_mat))
else:
    D_prc_mat = np.eye(U) / sqrt(U) # no precoding

# Step 3: Calculate per-user data rate
T_mat = B_mat * D_prc_mat   # Equivalent channel!
bl_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
bl_ami_pfm = np.zeros_like(snr_db_rng, dtype='float')
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)
    for u_id in range(U):
        sig_pow = np.abs(T_mat[u_id, u_id])**2
        
        ################################################ TEST!!!!
        N1 = N0 + (fnorm2(T_mat[u_id, :]) - sig_pow) / U
        
        bl_mi_pfm[u_id, snr_id] = log2(1 + sig_pow/N1)
        bl_ami_pfm[snr_id] += bl_mi_pfm[u_id, snr_id] / U
        
np.savetxt('bl_ami_pfm.csv', bl_ami_pfm, delimiter=',')

for u_id in range(U):
    plt.plot(snr_db_rng, bl_mi_pfm[u_id, :], '-', label='BL-user {0}'.format(u_id))

plt.plot(snr_db_rng, bl_ami_pfm, 'ko-', label='BL-mean')
sm_ami_pfm = np.loadtxt('sm_ami_pfm.csv', delimiter=',')
plt.plot(snr_db_rng, sm_ami_pfm, 'bx-', label='SM-mean')

plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Mutual Information (bits/s/Hz)')
plt.grid()
plt.show()
    





