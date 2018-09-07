"""
@Author:    Longzhuang He
@Descrp:    Multi-User SM MmWave Demonstration
"""

import numpy as np
from utils import *
import matplotlib.pyplot as plt

# For reproductivity
np.random.seed(23333)

"""
Basic configuration
"""
Nt      = 64
U       = 4
Nch     = 5
Nrf     = U
Nu_set  = [1, 2, 4, 8]
Nu_tot  = sum(Nu_set)

"""
Generate channel matrix
"""
gain_dec = 0.7
aoa_set, aod_set = gen_aoas_aods(U, Nch)
H_mat, P_mat, L_mat, Q_mat = \
gen_mmWave_chans(gain_dec, U, Nch, Nt, Nu_set, aoa_set, aod_set)

"""
Calculate single user mutual information
"""
snr_db_rng = np.linspace(-20, 10, 20)
mu_sel_rng = np.array([1, 2, 4])
mu_sel_res = np.zeros([len(snr_db_rng), U], 'int')
for snr_id, snr_db in enumerate(snr_db_rng):
    N0 = 10**(-snr_db/10)
    for u in range(U):
        Nu = Nu_set[u]
        Hu_mat = H_mat[sum(Nu_set[0:u]):sum(Nu_set[0:u])+Nu, :]

        mu_pfm = np.zeros(len(mu_sel_rng))
        for id_mu, mu in enumerate(mu_sel_rng):
            wu_set = np.ones(mu) / np.sqrt(U)

            # Till here!
            A_prc = np.mat(a_resp_mat(aod_set[u, 0:mu], Nt)) / np.sqrt(Nt)
            D_prc = np.mat(np.diag(wu_set))
            G_mat = Hu_mat * A_prc * D_prc
            mu_pfm[id_mu], _, _ = sm_mut_inf(G_mat, N0)
        
        mu_sel_res[snr_id, u] = mu_sel_rng[np.argmax(mu_pfm)]

"""
Calculate sum-rate
"""
sm_mi_pfm = np.zeros([U, len(snr_db_rng)])
for snr_id, snr_db in enumerate(snr_db_rng):
    print('{0}/{1}'.format(snr_id+1, len(snr_db_rng)))
    N0 = 10**(-snr_db/10)
    mu_set = mu_sel_res[snr_id, :]
    m_total = sum(mu_set)

    P_mat = np.mat(np.zeros([Nt, m_total], dtype='complex'))
    for u in range(U):
        mu = mu_set[u]
        A_prc = np.mat(a_resp_mat(aod_set[u, 0:mu], Nt))/np.sqrt(Nt)
        D_prc = np.mat(np.eye(mu))/np.sqrt(U)
        temp = A_prc * D_prc

        P_mat[:, sum(mu_set[0:u]):sum(mu_set[0:u])+mu] = A_prc * D_prc

    G_mat = H_mat * P_mat
    
    for u in range(U):
        nu = Nu_set[u]  # Receive antennas for User-u
        mu = mu_set[u]  # Virtual RF chain for User-u
        
        # Sub-channel matrix
        Gu = G_mat[sum(Nu_set[0:u]):sum(Nu_set[0:u])+nu, sum(mu_set[0:u]):sum(mu_set[0:u])+mu]
        N1 = N0 + (fnorm2(G_mat[sum(Nu_set[0:u]):sum(Nu_set[0:u])+nu, :]) - fnorm2(Gu)) / nu / (m_total-mu) * (U-1)
        sm_mi_pfm[u, snr_id], _, _ = sm_mut_inf(Gu, N1)

sm_ami_pfm = np.sum(sm_mi_pfm, axis=0) / U

"""
Plot the curves
"""
for u in range(U):
    plt.plot(snr_db_rng, sm_mi_pfm[u, :], 'o-', label='User {0}'.format(u))

plt.plot(snr_db_rng, sm_ami_pfm, 'ko-', label='Mean')
plt.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Mutual Information (bits/s/Hz)')
plt.grid()
plt.show()
