"""
@Author:    Longzhuang He
@Descrp:    Utility functions
"""

import numpy as np

def error(s):
    raise(Exception(s))

def c_randn_2d(d1, d2):
    # Generate 2-d standard normal distributed matrix (i.i.d.)
    res = (np.random.randn(d1, d2) + 1j*np.random.randn(d1, d2)) * np.sqrt(1/2)
    return res

def gen_aoas_aods(U, Nch):
    # Generate AoAs & AoDs for each user
    # Angle range: [-pi/2, pi/2]
    phi_aod_set = np.zeros(shape=[U, Nch])
    phi_aod_set[:, 0] = np.array(range(1, U+1, 1))/(U+1)
    phi_aod_set[:, 1:] = np.random.rand(U, Nch-1)
    phi_aod_set = -1/2 + phi_aod_set
    theta_aoa_set = -1/2 + np.random.rand(U, Nch)

    return theta_aoa_set, phi_aod_set

def a_resp_vec(phi, n):
    if phi < -0.5 or phi > 0.5:
        error('!!!!')

    # Generate n-dim antenna-response vector
    temp = np.array(range(n), 'float') - (n-1)/2
    a_vec = np.exp(-1j*2*np.pi*phi*temp)
    return a_vec

def a_resp_mat(phi_set, n):
    if np.array(phi_set).shape == ():
        return a_resp_vec(phi_set, n)
    
    a_mat =np.zeros([n, len(phi_set)], dtype='complex')
    for i, phi in enumerate(phi_set):
        a_mat[:, i] = a_resp_vec(phi, n)
    return a_mat

def gen_mmWave_chans(gain_dec, U, Nch, Nt, Nu_set, aoa_set, aod_set):
    # L_mat (channel gains)
    gain_vars = np.array([gain_dec**w for w in range(Nch)])
    path_gain_set = np.dot(c_randn_2d(U, Nch), np.diag(np.sqrt(gain_vars)))
    L_mat = np.mat(np.diag(np.reshape(path_gain_set, U*Nch)))

    # Q_mat
    Q_mat = np.zeros([Nt, U*Nch], dtype='complex')
    for u in range(U):
        Q_mat[:, Nch*u:Nch*(u+1)] = a_resp_mat(aod_set[u, :], Nt)
    Q_mat = np.mat(Q_mat)

    # P_mat
    Nu_all = np.sum(Nu_set)
    P_mat = np.zeros([Nu_all, U*Nch], dtype='complex')
    for u in range(U):
        Nu = Nu_set[u]
        Pu = a_resp_mat(aoa_set[u, :], Nu)

        r_st_idx = int(np.sum(Nu_set[0:u]))
        r_ed_idx = int( r_st_idx + Nu)

        c_st_idx = int(Nch*u)
        c_ed_idx = int(Nch*(u+1))

        P_mat[r_st_idx:r_ed_idx, c_st_idx:c_ed_idx] = Pu
    
    P_mat = np.mat(P_mat)

    H_mat = P_mat * L_mat * Q_mat.H
    return H_mat, P_mat, L_mat, Q_mat
    
def sm_mut_inf(G, N0):
    if ~isinstance(G, np.matrix):
        G = np.mat(G)
    
    N, M = G.shape
    sym_dm_mi = np.mean([np.log2(np.real(np.linalg.det(np.eye(N) + G[:, col_id]*G[:, col_id].H / N0))) for col_id in range(M)])

    sig_mat_set = dict()
    for id_m in range(M):
        sig_mat_set[id_m] = np.mat(N0*np.eye(N) + G[:, id_m]*G[:, id_m].H)

    spt_dm_mi_theo = 0
    if M > 1:
        for id_n in range(M):
            temp = 0
            for id_t in range(M):
                temp += np.real(np.linalg.det(sig_mat_set[id_n])) / np.real(np.linalg.det(sig_mat_set[id_n] + sig_mat_set[id_t]))
            
            spt_dm_mi_theo += np.log2(temp)
        spt_dm_mi_theo = np.log2(M) - N - 1/M * spt_dm_mi_theo

    return sym_dm_mi+spt_dm_mi_theo, sym_dm_mi, spt_dm_mi_theo

def fnorm2(M):
    return np.real(np.sum(np.real(np.power(np.abs(M), 2.0))))