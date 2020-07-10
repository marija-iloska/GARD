function [fs_ard] = ard_f(A, I, I0, K, A_init, C_est, mu_x, sig_x, a)

% Gibbs ARD
for prior = 1:length(a)
    tic
    [A_s] = gibbs_ard(I, I0, K, A_init, C_est, mu_x, sig_x, a(prior));
    toc

    A_ard = mode(A_s,3);

    [~,~, fs_ard(prior)] = adj_eval(A, A_ard);

end

end