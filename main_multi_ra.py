import numpy as np
import matplotlib.pyplot as plt
from utils import *
from numpy import pi, log2, real, exp, sqrt
from numpy.linalg import det, cholesky as chol

np.random.seed(13)

# Basic configuration
Nl = 2
Nr = 4
# phi_set = np.random.rand(Nl) * 1/2
phi_set = np.array([0.3, 0.4])
snr_db = 30
N0 = 10**(-snr_db/10)

g_phi = (np.sin(pi*Nr*(phi_set[1]-phi_set[0])) / np.sin(pi*(phi_set[1]-phi_set[0])))**2
P = np.matrix(np.zeros(shape=[Nr, Nl], dtype='complex'))
for i in range(Nl):
    P[:, i] = np.matrix(a_resp_vec(phi_set[i], Nr)).T

# For spatial domain MI calculation
Nsample = int(1e3)
cgss_vec_set = c_randn_2d(Nr, Nsample)

# Calculate MI
w_rng = np.linspace(1e-2, 1-1e-2, 50)
sa_pfm = np.zeros_like(w_rng)
sm_pfm = np.zeros_like(w_rng)
sm_pfm_aprx = np.zeros_like(w_rng)

for w_id, w0 in enumerate(w_rng):
    w1 = 1 - w0
    w_set = [w0, w1]
    A = np.diag(np.array(np.sqrt([w0, w1])))
    H = P * A

    # Calculate single-antenna MI
    sa_pfm[w_id] = log2(1 + w0/N0)

    # Calculate spatial-modulation MI
    sym_dm_mi = 1/2 * (log2(1 + w0/N0) + log2(1 + w1/N0))

    ant_dm_mi = 0
    sig_mat_0 = N0*np.eye(Nr) + H[:, 0] * H[:, 0].H
    sig_mat_1 = N0*np.eye(Nr) + H[:, 1] * H[:, 1].H
    for n in range(2):
        sig_mat_n = N0*np.eye(Nr) + H[:, n] * H[:, n].H
        zero_mu = np.zeros(shape=[Nr, 1], dtype='float')
        
        sig_mat_n_half = chol(sig_mat_n)
        y_set = sig_mat_n_half * cgss_vec_set

        temp_set = log2(c_gauss_2d(y_set, zero_mu, sig_mat_n)*2 / (c_gauss_2d(y_set, zero_mu, sig_mat_0) + c_gauss_2d(y_set, zero_mu, sig_mat_1)))
        ant_dm_mi += 1/2 * np.mean(temp_set)

    ant_dm_mi_apprx = 1 - log2(exp(1))/2 * (2+1/N0) / ((1+w0/(2*N0))*(1+w1/(2*N0)) - w0*w1*g_phi/(4*N0**2 * Nr**2))
    
    sm_pfm[w_id] = sym_dm_mi + ant_dm_mi
    sm_pfm_aprx[w_id] = sym_dm_mi + ant_dm_mi_apprx

plt.plot(w_rng, sa_pfm, 'b-', label='single-ant')
plt.plot(w_rng, sm_pfm, 'r-', label='sm')
plt.plot(w_rng, sm_pfm_aprx, 'k--', label='sm apprx')
plt.legend()
plt.grid()
plt.show()


# For test
# alpha = (1-g_phi/(Nr**2))/(2*N0**2)/(2+1/N0)
# y = [5*alpha*x**3 - 9*alpha*x**2 + (3+4*alpha)*x - 4 for x in w_rng]
# # y = [5+2/(alpha*x*(1-x)-1)-4/x for x in w_rng]
# # y = [2*exp(-(2+1/N0)/(2+1/N0 + (1-g_phi/(Nr**2))/(2*N0**2) * w0*(1-w0))) - sqrt((1+w0/N0)/(1+(1-w0)/N0)) for w0 in w_rng]
# plt.plot(w_rng, y, 'bo-')
# plt.grid()
# plt.show()



