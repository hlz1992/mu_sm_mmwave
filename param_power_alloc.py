import numpy as np
from utils import *
from numpy import pi, log2, real, exp, sqrt, abs, sum, mean, diag, power
from numpy.linalg import det, cholesky as chol
import matplotlib.pyplot as plt

# np.random.seed(2333)

Nch = 10     # No. channel paths
Nt = 64     # No. transmit antennas
U = 1       # No. users
Nr_set = np.array([8])[0:U]     # Users' receive antennas
Nr_total = sum(Nr_set)
snr_db = 10
N0 = 10**(-snr_db/10)
dec_fac = 0.7   # Path gain decaying factor

aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(dec_fac, U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

Mu = 2
pow_alloc_rng = np.linspace(1e-3, 1-1e-3, 20)
sm_pfm = np.zeros(shape=pow_alloc_rng.shape)
for id_pow in range(len(pow_alloc_rng)):
    pow_alloc = pow_alloc_rng[id_pow]

    wu_set = sqrt(np.array([pow_alloc, 1-pow_alloc])) / sqrt(U)

    print(wu_set)

    A_prc = np.mat(a_resp_mat(aod_set[0, 0:Mu], Nt))/sqrt(Nt)
    D_prc = np.mat(diag(wu_set))
    P_mat = A_prc * D_prc

    G_mat = H_mat * P_mat
    sm_pfm[id_pow], _, _ = sm_mut_inf(G_mat, N0)

plt.plot(pow_alloc_rng, sm_pfm, 'o-')
plt.grid()
plt.xlabel('power allocation')
plt.ylabel('SM MI (bits/s/Hz)')
plt.show()

