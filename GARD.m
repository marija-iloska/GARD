clear all
clc


%% SETTINGS for generating data
dim_y = 8; var_u =1;
dim_x=dim_y^2;
p_s = 0.7; p_ns = 0.3;
T = 100;
a = 0.5:0.25:2;


%% GIBBS SAMPLER: ARD prior

I = 2000;                       % Gibbs iterations
I0 = 1000;                      % Gibbs burn-in
K = 2;                          % Thinning parameter
A_init = ones(dim_y, dim_y);    % Initial adjacency matrix
R=100;



%% Bayesian Ridge Regression

%Consider a Gaussian prior
mu_0 = zeros(dim_x, 1);
var_0 = 1; sig_0 = var_0*eye(dim_x);

tic
parfor run = 1:R
    % Generating data and matrices
    [A, C, y, dim_x] = generate_mat(T, dim_y, p_s, p_ns, var_u);
    
    
    % Bayesian ridge regression for C
    [mu_c, sig_c, mu_x, sig_x] = mn_conjugate_var(y, var_u, mu_0, sig_0);
    
    % Convert to an estimate of the coefficient matrix
    C_est = reshape(mu_c, dim_y, dim_y);
    
    % Compute MSE in the coefficients
    MSE = sum(sum((C-C_est).^2))/dim_x;
    
    [fs_ard] = ard_f(A, I, I0, K, A_init, C_est, mu_x, sig_x, a)
    
    fs(run,:) = fs_ard;
    
end
toc

fs_final = mean(fs,1);
col = [0.5, 0 0];
plot(a, fs_final, 'Color', col,'linewidth', 3)
title('ARD prior', 'fontsize', 18)
xlabel('\alpha prior', 'fontsize', 18)
ylabel('f-score', 'fontsize', 18)



