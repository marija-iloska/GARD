function [A_s] = gibbs_ard(iter_num, burn_in, thin_rate, A_init, C, mu, sig, a)

% Obtain dimensions
dim_y = length(C(:, 1));
b = dim_y;

% Initialize arrays to store samples
A_s = zeros(dim_y, dim_y, iter_num+1);

% Set the initial states of the Markov chains
A_s(:, :, 1) = A_init;
A_old = A_s(:, :, 1);

% For loop for the Gibbs
for i = 2 :iter_num+1

    for j = 1:dim_y       
        for k = 1:dim_y
            %b = max(1, sum(A_old(:,k)));
            b =  max(1, sum(A_old(j,:)));
            %b = max(1, sum(A_old(j,:) + A_old(:,k)')-A_old(j,k));
            gamma(j,k) = gamrnd(a + 0.5, A_old(j,k)*0.5 + b,1,1);

            % Conside that A(j, k)=1
            A_old(j,k) = 1;
            C_temp = C.*A_old;
            log_pa1 = log(normpdf(A_old(j,k),0, 1/gamma(j,k))) + logmvnpdf(C_temp(:)', mu', sig);

            % Conside that A(j, k)=0
            A_old(j,k) = 0;
            C_temp = C.*A_old;
            log_pa0 = log(normpdf(A_old(j,k), 0, 1/gamma(j,k))) + logmvnpdf(C_temp(:)', mu', sig);
            
            % Sample the topology
            pa0 = exp(log_pa0 - max([log_pa0, log_pa1]));
            pa1 = exp(log_pa1 - max([log_pa0, log_pa1]));
            prob_1 = pa1/(pa1 + pa0);
            
            A_old(j,k) = rand <prob_1;     
        end
        
    end
    
    A_s(:, :, i) = A_old;
    
end

% Apply burn-in
A_s = A_s(:, :, burn_in+1:thin_rate:iter_num+1);

end

