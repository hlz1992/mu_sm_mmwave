function a_vec = a_resp_vec(phi, n)
temp = [0:n-1].' - (n-1)/2;
a_vec = exp(-1j * 2*pi * phi * temp) / sqrt(n);
end