#######################################################################
# 
#   @author: Longzhuang He
#   @descrp: SM wwave system with digital precoding
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

# Step 1: Determine per-user hyper-RF-chain num
mu_sel_rng = np.array([1, 2, 4])
mu_sel_res = np.zeros(shape=[len(snr_db_rng), U], dtype='int')
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)
    for u_id in range(U):
        Nu = Nr_set[u_id]
        Hu_mat = H_mat[sum(Nr_set[0:u_id]):sum(Nr_set[0:u_id])+Nu, :]

        mu_pfm = np.zeros(shape=[len(mu_sel_rng)])
        for mu_id in range(len(mu_sel_rng)):
            mu = mu_sel_rng[mu_id]
            wu_set = np.ones(shape=[mu]) / sqrt(U)
            
            A_prc = np.mat(a_resp_mat(aod_set[u_id, 0:mu], Nt))/sqrt(Nt)
            D_prc = np.mat(diag(wu_set))
            W_mat = np.mat(a_resp_mat(aoa_set[u_id, 0:mu], Nu))/sqrt(Nu)
            
            G_mat = Hu_mat * A_prc * D_prc
            B_mat = W_mat.H * G_mat
            
            mu_pfm[mu_id], xxx, yyyy = sm_mut_inf(B_mat, N0)
        mu_sel_res[snr_id, u_id] = mu_sel_rng[argmax(mu_pfm)]

# Step 2: Calculate sum rate
sm_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
sm_ami_pfm = np.zeros_like(snr_db_rng, dtype='float')
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)
    mu_set = mu_sel_res[snr_id, :]
    m_total = sum(mu_set)
    
    # Determine receiver combining scheme
    W_mat = np.mat(np.zeros([Nr_total, m_total], dtype='complex'))
    for u_id in range(U):
        Nu = Nr_set[u_id]
        st_idx_1 = sum(Nr_set[0:u_id])
        ed_idx_1 = st_idx_1 + Nu
        
        mu = mu_set[u_id]
        st_idx_2 = sum(mu_set[0:u_id])
        ed_idx_2 = st_idx_2 + mu
        
        Wu = np.mat(a_resp_mat(aoa_set[u_id, 0:mu], Nu))/sqrt(Nu)
        W_mat[st_idx_1:ed_idx_1, st_idx_2:ed_idx_2] = Wu
    
    # Determine transmitter analog precoding & power allocation scheme
    S_mat = np.mat(np.zeros(shape=[Nt, m_total], dtype='complex'))
    for u_id in range(U):
        mu = mu_set[u_id]
        A_prc = np.mat(a_resp_mat(aod_set[u_id, 0:mu], Nt))/sqrt(Nt)
        D_prc = np.mat(np.eye(mu)) / sqrt(U)    # Power allocation on different super-RF chains (each user has 1/U power)

        S_mat[:, sum(mu_set[0:u_id]):sum(mu_set[0:u_id])+mu] = A_prc * D_prc
    
    # Equivalent channel
    B_mat = W_mat.H * H_mat * S_mat
    for u_id in range(U):
        nu = Nr_set[u_id]
        mu = mu_set[u_id]
        
        st_idx = sum(mu_set[0:u_id])
        ed_idx = st_idx + mu
        Bu = B_mat[st_idx:ed_idx, st_idx:ed_idx]
        
        ################################################ TEST!!!!
        N1 = N0 + (fnorm2(B_mat[st_idx:ed_idx, :]) - fnorm2(Bu)) / (m_total-mu) * (U-1)/U
        sm_mi_pfm[u_id, snr_id] = sa_mut_inf(Bu, N1)
        sm_ami_pfm[snr_id] += sm_mi_pfm[u_id, snr_id] / U

np.savetxt('sm_ami_pfm.csv', sm_ami_pfm, delimiter=',')

for u_id in range(U):
    plt.plot(snr_db_rng, sm_mi_pfm[u_id, :], '-', label='SM-user {0}'.format(u_id))

plt.plot(snr_db_rng, sm_ami_pfm, 'bx-', label='SM-mean')

bl_ami_pfm = np.loadtxt('bl_ami_pfm.csv', delimiter=',')
plt.plot(snr_db_rng, bl_ami_pfm, 'ko-', label='BL-mean')

plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Mutual Information (bits/s/Hz)')
plt.grid()
plt.show()


