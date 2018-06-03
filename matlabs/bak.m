close all, clear all, clc;

w_rng = linspace(0, 1, 20);
sa_pfm = 0 * w_rng;
sm_pfm = 0 * w_rng;

snr_db = 20;
N0 = 10^(-snr_db/10);

rnd_num = 1e3;
wgss_set = (randn(1, rnd_num)+1j*randn(1, rnd_num)) * sqrt(1/2);

for w_id = 1:length(w_rng)
    w1 = w_rng(w_id);
    w2 = 1 - w1;
    w_set = [w1, w2];
    
    sa_pfm(w_id) = log2(1+w1/N0);
    
    temp = 0;
    temp_set = 0 * wgss_set;
    for n = [1, 2]
        temp_set = 0 * temp_set;
        wgss_set_temp = wgss_set * sqrt(N0 + w_set(n));
        for id_y = 1:length(wgss_set_temp)
            y = wgss_set_temp(id_y);
            temp_set(id_y) = real(log2(c_gauss(y, 0, N0+w_set(n))/(c_gauss(y, 0, N0+w_set(1)) + c_gauss(y, 0, N0+w_set(2)))*2));
        end
        temp = temp + 1/2 * mean(temp_set);
    end
    sm_pfm(w_id) = temp + 1/2*(log2(1+w1/N0) + log2(1+w2/N0));
end

figure; hold on;
plot(w_rng, sa_pfm, 'bo-');
plot(w_rng, sm_pfm, 'rx-');
legend('single antenna', 'spatial modulation')

