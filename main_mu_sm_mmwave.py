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

# Generate channel matrix
dec_fac = 0.8   # Path gain decaying factor
aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(dec_fac, U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

snr_db_rng = np.linspace(-20, 10, 20)    # SNR range (in dB)

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
            P_mat = A_prc * D_prc
            G_mat = Hu_mat * P_mat
            mu_pfm[mu_id], _, _ = sm_mut_inf(G_mat, N0)
        mu_sel_res[snr_id, u_id] = mu_sel_rng[argmax(mu_pfm)]

# Step 2: Calculate sum rate
sm_mi_pfm = np.zeros(shape=[U, len(snr_db_rng)])
sm_ami_pfm = np.zeros_like(snr_db_rng)
for snr_id in range(len(snr_db_rng)):
    N0 = 10**(-snr_db_rng[snr_id]/10)
    mu_set = mu_sel_res[snr_id, :]
    m_total = sum(mu_set)
    
    P_mat = np.mat(np.zeros(shape=[Nt, m_total], dtype='complex'))
    for u_id in range(U):
        mu = mu_set[u_id]
        A_prc = np.mat(a_resp_mat(aod_set[u_id, 0:mu], Nt))/sqrt(Nt)
        D_prc = np.mat(np.eye(mu)) / sqrt(U)
        temp = A_prc * D_prc
        P_mat[:, sum(mu_set[0:u_id]):sum(mu_set[0:u_id])+mu] = A_prc * D_prc
    
    G_mat = H_mat * P_mat
    for u_id in range(U):
        nu = Nr_set[u_id]
        mu = mu_set[u_id]
        Gu = G_mat[sum(Nr_set[0:u_id]):sum(Nr_set[0:u_id])+nu, sum(mu_set[0:u_id]):sum(mu_set[0:u_id])+mu]
        N1 = N0 + (fnorm2(G_mat[sum(Nr_set[0:u_id]):sum(Nr_set[0:u_id])+nu, :]) - fnorm2(Gu)) / nu / (m_total-mu)
        sm_mi_pfm[u_id, snr_id] = sa_mut_inf(Gu, N1)
        sm_ami_pfm[snr_id] += sm_mi_pfm[u_id, snr_id] / U

for u_id in range(U):
    plt.plot(snr_db_rng, sm_mi_pfm[u_id, :], '-', label='SM-user {0}'.format(u_id))

plt.plot(snr_db_rng, sm_ami_pfm, 'ko-', label='SM-mean')
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Mutual Information (bits/s/Hz)')
plt.grid()
plt.show()

