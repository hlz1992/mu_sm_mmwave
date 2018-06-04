import numpy as np
import matplotlib.pyplot as plt

def error(s):
    # Print error message to halt the program
    raise(Exception(str(s)))
    pass

def c_gauss_2d(x, mu_vec, sig_mat):
    # Generate complex-valued multivariate Gaussian PDF using matrix-like x and col-vector-like mu_vec & sig_mat
    if ~isinstance(x, np.matrix):
        x = np.matrix(x)

    if ~isinstance(mu_vec, np.matrix):
        mu_vec = np.matrix(mu_vec)
        if mu_vec.shape[0] == 1:
            mu_vec = mu_vec.T

    if ~isinstance(sig_mat, np.matrix):
        sig_mat = np.matrix(sig_mat)

    ndim, nsam = x.shape
    res = 1/(np.pi**ndim * np.real(np.linalg.det(sig_mat))) * np.exp(-np.real(np.diag((x-mu_vec).H * sig_mat.I * (x-mu_vec))))
    return res

def c_gauss_1d(x, mu_vec, sig_mat):
    # Generate complex-valued multivariate Gaussian PDF using col-vector-like x, mu_vec & sig_mat
    if ~isinstance(x, np.matrix):
        x = np.matrix(x)

    if ~isinstance(mu_vec, np.matrix):
        mu_vec = np.matrix(mu_vec)

    if ~isinstance(sig_mat, np.matrix):
        sig_mat = np.matrix(sig_mat)

    res = 1/(np.pi**len(x) * np.real(np.linalg.det(sig_mat))) * np.exp(-np.real((x - mu_vec).H * sig_mat.I * (x - mu_vec)))
    return float(res)

def c_gauss_0d(x, mu, sigma):
    # Generate complex-valued Gaussian PDF using scalar x, mu & sigma
    res = 1/(np.pi * sigma) * np.exp(-np.abs(x-mu)**2 / sigma)
    return res

def c_randn(n):
    # Generate 1-d standard normal distributed vector (i.i.d.)
    res = np.random.randn(n)*np.sqrt(1/2) + 1j * np.random.randn(n)*np.sqrt(1/2)
    return res

def c_randn_2d(ndim, nsample):
    # Generate 2-d standard normal distributed matrix (i.i.d.)
    res = (np.random.randn(ndim, nsample) + 1j*np.random.randn(ndim, nsample)) * np.sqrt(1/2)
    return res

def a_resp_vec(phi, n):
    # Generate n-dim antenna-response vector
    temp = np.array(range(n), 'float') - (n-1)/2
    # a_vec = np.exp(-1j * 2*np.pi * phi * temp) / np.sqrt(n)
    a_vec = np.exp(-1j * 2*np.pi * phi * temp)
    return a_vec

def a_resp_mat(phi_set, n):
    a_mat = np.zeros(shape=[n, len(phi_set)], dtype='complex')
    for i in range(len(phi_set)):
        a_mat[:, i] = a_resp_vec(phi_set[i], n)
    return a_mat

def gen_aoas_aods(U, Nch):
    # Generate AoAs & AoDs for each user
    phi_aod_set = np.zeros(shape=[U, Nch])
    phi_aod_set[:, 0] = np.array(range(1, U+1, 1))/(U+1)
    phi_aod_set[:, 1:] = np.random.rand(U, Nch-1)
    phi_aod_set = -np.pi/2 + np.pi*phi_aod_set

    theta_aoa_set = np.random.rand(U, Nch)
    theta_aoa_set = -np.pi/2 + np.pi*theta_aoa_set

    return theta_aoa_set, phi_aod_set

def gen_mu_mmwave_chans(U, Nch, Nt, Nr_set, aoa_set, aod_set):
    # Generate multi-user mmWave-MIMO channel matrix
    gain_vars = np.array([0.1**w for w in range(Nch)])

    path_gain_set = np.dot(c_randn_2d(U, Nch), np.diag(np.sqrt(gain_vars)))
    # path_gain_set = np.ones(shape=[U, Nch])
    
    Lam_mat = np.mat(np.diag(np.reshape(path_gain_set, U*Nch)))

    # Q_mat
    Q_mat = np.zeros(shape=[Nt, U*Nch], dtype='complex')
    for u_id in range(U):
        Q_mat[:, Nch*u_id:Nch*(u_id+1)] = a_resp_mat(aod_set[u_id, :], Nt)
    Q_mat = np.mat(Q_mat)
        
    # P_mat
    Nr_all = np.sum(Nr_set)
    P_mat = np.zeros(shape=[Nr_all, U*Nch], dtype='complex')
    for u_id in range(U):
        Nu = Nr_set[u_id]
        Pu = a_resp_mat(aoa_set[u_id, :], Nu)

        r_st_idx = np.sum(Nr_set[0:u_id])
        r_ed_idx = np.sum(Nr_set[0:u_id])+Nu

        c_st_idx = Nch*u_id
        c_ed_idx = Nch*(u_id+1)

        P_mat[r_st_idx:r_ed_idx, c_st_idx:c_ed_idx] = Pu

    P_mat = np.mat(P_mat)

    H_mat = P_mat * Lam_mat * Q_mat.H

    return H_mat, P_mat, Lam_mat, Q_mat
        
def fnorm2(M):
    return np.sum(np.real(np.power(np.abs(M), 2)))

def sa_mut_inf(h, N0):
    # Calculate single-TA multi-RA system's mutual information
    return np.log2(np.real(np.linalg.det(np.eye(len(h)) + np.mat(h) * np.mat(h).H / N0)))

def sm_mut_inf(G, N0):
    if ~isinstance(G, np.matrix):
        G = np.mat(G)

    N, M = G.shape
    sym_dm_mi = np.mean([np.log2(np.real(np.linalg.det(np.eye(N) + G[:, col_id]*G[:, col_id].H / N0))) for col_id in range(M)])

    sig_mat_set = dict()
    for id_m in range(M):
        sig_mat_set[id_m] = np.mat(N0*np.eye(N) + G[:, id_m]*G[:, id_m].H)
    
    spt_dm_mi_theo = 0
    for id_n in range(M):
        temp = 0
        for id_t in range(M):
            temp += np.real(np.linalg.det(sig_mat_set[id_n])) / np.real(np.linalg.det(sig_mat_set[id_n] + sig_mat_set[id_t]))
        
        spt_dm_mi_theo += np.log2(temp)
    spt_dm_mi_theo = np.log2(M) - N - 1/M * spt_dm_mi_theo

    return sym_dm_mi+spt_dm_mi_theo, sym_dm_mi, spt_dm_mi_theo


    








