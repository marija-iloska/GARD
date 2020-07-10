function [A_s] = gibbs_ard_ctest(iter_num, burn_in, thin_rate, A_init, a, lambda_init, y, var_u)

% Obtain dimensions
dim_y = length(A_init(:, 1));
b = dim_y;
dim_x = dim_y^2;

% Initialize arrays to store samples
A_s = zeros(dim_y, dim_y, iter_num+1);

% Set the initial states of the Markov chains
A_s(:, :, 1) = A_init;
A_old = A_s(:, :, 1);

gamma_0 = ones(dim_y, dim_y, iter_num+1);

lambda_s = zeros(iter_num+1, 1);
lambda_s(1) = lambda_init;


%Consider a Gaussian prior
mu_0 = zeros(dim_x, 1);


% For loop for the Gibbs
for i = 2 :iter_num+1
    
    
    % SAMPLING C__________________________________________________
    
    
    % Obtain the posterior of the vectorized matrix (and likelihood
    % counterparts)
    gamma = gamma_0(dim_y, dim_y, i-1);
    temp = gamma(:);
    var_0 = 1./temp; sig_0 = var_0.*eye(dim_x);
    
    [mu_c, ~, mu, sig] = mn_conjugate_var(y, var_u, mu_0, sig_0);

    % Convert to an estimate of the coefficient matrix
    C = reshape(mu_c, dim_y, dim_y);
    
    
    % SAMPLING A and Gamma ______________________________________________
    for j = 1:dim_y       
        for k = 1:dim_y
            b = max(1, sum(A_old(:,k)));
            gamma_0(j,k, i) = gamrnd(a + 0.5, (C(j,k)^2)*0.5 + b,1,1);

            A_old(j,k) = 1;
            C_temp = C.*A_old; 
            log_pa1 = -lambda_s(i-1)*sum(A_old(j,:)) + logmvnpdf(C_temp(:)', mu', sig);
           
            % Conside that A(j, k)=0
            A_old(j,k) = 0;
            C_temp = C.*A_old; 
            log_pa0 = -lambda_s(i-1)*sum(A_old(j,:)) + logmvnpdf(C_temp(:)', mu', sig);
           
            % Sample the topology
            pa0 = exp(log_pa0-max([log_pa0, log_pa1]));
            pa1 = exp(log_pa1-max([log_pa0, log_pa1]));
            prob_1 = pa1/(pa1+pa0);
            A_old(j,k) = rand<prob_1;
            
            A_old(j,k) = rand <prob_1;     
        end
        
    end
    lambda_s(i) = gamrnd(dim_y+1, gamma + dim_y*sum(A_old(j,:)));
    
    
    A_s(:, :, i) = A_old;
    
end

% Apply burn-in
A_s = A_s(:, :, burn_in+1:thin_rate:iter_num+1);

end

