function res = c_gauss(x, mu, sigma)
res = 1/(pi^length(x)*det(sigma))*...
    exp(-(x-mu)' * inv(sigma) * (x-mu));

end