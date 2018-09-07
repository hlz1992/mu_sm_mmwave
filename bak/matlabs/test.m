close all, clear all, clc;

Nl = 2;
Nr = 2;
% phi_set = rand(1, Nl) * 1/2;
phi_set = [0.5, 0.5];
snr_db = 20;
N0 = 10^(-snr_db/10);

P = zeros(Nr, Nl);
for i = [1:Nl]
    P(:, i) = a_resp_vec(phi_set(i), Nr);
end

Nsample = 1e3;
cgss_vec_set = (randn(Nr, Nsample) + 1j*randn(Nr, Nsample)) * sqrt(1/2);

w_rng = linspace(0, 1, 20);
sa_pfm = 0 * w_rng;
sm_pfm = 0 * w_rng;

for w_id = 1:length(w_rng)
    w1 = w_rng(w_id);
    w2 = 1 - w1;
    
    A = sqrt(diag([w1, w2].'));
    H = P * A;
    
    sa_pfm(w_id) = log2(real(det(eye(Nr) + 1/N0 * H(:, 1) * H(:, 1)')));
    
    sym_dm_mi = 1/2 * (log2(real(det(eye(Nr) + 1/N0 * H(:, 1) * H(:, 1)'))) + ...
        log2(real(det(eye(Nr) + 1/N0 * H(:, 2) * H(:, 2)'))));
    
    ant_dm_mi = 0;
    sig_mat_0 = N0*eye(Nr) + H(:, 1) * H(:, 1)';
    sig_mat_1 = N0*eye(Nr) + H(:, 2) * H(:, 2)';
    for n = 1:2
        sig_mat_n = N0*eye(Nr) + (H(:, n) * H(:, n)');
        temp = real(diag(sig_mat_n));
        sig_mat_n = sig_mat_n - diag(diag(sig_mat_n)) + diag(temp);
        
        sig_mat_n_half = chol(sig_mat_n)';
        y_set = sig_mat_n_half * cgss_vec_set;
        
        temp_set = zeros(1, Nsample);
        for id_sample = 1:Nsample
            A = c_gauss(y_set(:, id_sample), zeros(Nr, 1), sig_mat_0);
            B = c_gauss(y_set(:, id_sample), zeros(Nr, 1), sig_mat_1);
            temp_set(id_sample) = log2(...
                real(c_gauss(y_set(:, id_sample), zeros(Nr, 1), sig_mat_n)*2/...
                (A + B)));
        end
        ant_dm_mi = ant_dm_mi + 1/2 * mean(temp_set);
    end
    
    sm_pfm(w_id) = sym_dm_mi + ant_dm_mi;
end

plot(w_rng, sa_pfm, 'bo-');
hold on;
plot(w_rng, sm_pfm, 'rx-');

