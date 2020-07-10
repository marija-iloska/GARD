function [mu, sig, mu_x, sig_x] = mn_conjugate_var(y, var_y, mu_0, sig_0)

% Obtain dimension of the observations
dim_y = length(y(:, 1));     T = length(y(1, :));


% Obtain Matrix Notrmal parameters of likelihood
M = (y(:, 1:T-1)*y(:, 1:T-1)')\(y(:, 1:T-1)*y(:, 2:T)')';
U = var_y*inv(y(:, 1:T-1)*y(:, 1:T-1)');
V = eye(dim_y);


% Convert it to the vector form
mu_x = M(:);
sig_x = kron(V, U);
dim_x = dim_y^2;


% Compute the posterior of vectorized matrix in closed form 
sig = inv(inv(sig_0)+inv(sig_x));
mu = sig*(inv(sig_0)*mu_0+inv(sig_x)*mu_x);

end

