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
dec_fac = 1.0   # Path gain decaying factor

aoa_set, aod_set = gen_aoas_aods(U, Nch)            # AoAs & AoDs for each user and each path
H_mat, P_mat, Lam_mat, Q_mat = gen_mu_mmwave_chans(dec_fac, U, Nch, Nt, Nr_set, aoa_set, aod_set)    # Multi-user mmWave channel matrix

Mu_rng = np.array([1, 2, 4, 8])
sm_pfm = np.zeros(shape=[len(Mu_rng)])
for id_Mu in range(len(Mu_rng)):
    Mu = Mu_rng[id_Mu]
    wu_set = np.ones(shape=[Mu]) / sqrt(U)

    A_prc = np.mat(a_resp_mat(aod_set[0, 0:Mu], Nt))/sqrt(Nt)
    D_prc = np.mat(diag(wu_set))
    P_mat = A_prc * D_prc

    G_mat = H_mat * P_mat
    sm_pfm[id_Mu], _, _ = sm_mut_inf(G_mat, N0)

print(sm_pfm)
plt.stem(sm_pfm)
plt.show()
