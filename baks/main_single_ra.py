import numpy as np
import matplotlib.pyplot as plt
from utils import *

np.random.seed(13)

c_gauss_vec = np.vectorize(c_gauss_0d)

w_rng = np.linspace(0, 1, 20)
sa_pfm = np.zeros_like(w_rng)   # single antenna
sm_pfm = np.zeros_like(w_rng)   # spatial modulation

snr_db = 20
N0 = 10**(-snr_db/10)

rnd_num = int(1e3)
wgss_set = c_randn(rnd_num)

for w_id, w1 in enumerate(w_rng):
    w2 = 1 - w1
    w_set = (w1, w2)

    # Calculate single antenna
    sa_pfm[w_id] = np.log2(1 + w1/N0)

    # Calculate spatial modulation
    temp = 0
    temp_set = np.zeros_like(wgss_set)
    for n in range(2):
        temp_set = 0 * temp_set
        wgss_set_temp = wgss_set * np.sqrt(N0 + w_set[n])

        temp_set = np.real(np.log2(c_gauss_vec(wgss_set_temp, 0, N0+w_set[n])/(c_gauss_vec(wgss_set_temp, 0, N0+w_set[0]) + c_gauss_vec(wgss_set_temp, 0, N0+w_set[1]))*2))

        temp += 1/2 * np.mean(temp_set)
    
    sm_pfm[w_id] = 1/2*(np.log2(1 + w_set[0]/N0) + np.log2(1 + w_set[1]/N0)) + temp

plt.plot(w_rng, sa_pfm, 'bo-', label='single antenna')
plt.plot(w_rng, sm_pfm, 'rx-', label='spatial modulation')
plt.legend()
plt.show()
