function [samples, weights] = generate_samples(positions, orientations, sigma_pos, sigma_ang, N)
% Generates N samples per person with Gaussian noise
K = size(positions, 1);
samples = zeros(K * N, 3);
weights = zeros(K * N, 1);

for k = 1:K
    mu = [positions(k, :) orientations(k)];
    Sigma = diag([sigma_pos^2, sigma_pos^2, sigma_ang^2]);

    tmp_samples = mvnrnd(mu, Sigma, N);
    samples((k-1)*N+1:k*N, :) = tmp_samples;
    weights((k-1)*N+1:k*N) = mvnpdf(tmp_samples, mu, Sigma);
end
end
