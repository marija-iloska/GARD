function [A, C, y, dim_x] = generate_mat(T, dim_y, p_s, p_ns, var_u)

% Get dimension of vectorized matrix
dim_x = dim_y^2;

% Initialize coefficient and aadjacency matrices
C = rand(dim_y, dim_y)*1-0.5;
A = ones(dim_y, dim_y);

% Generate Matrices
for i = 1:dim_y
    cond = true;
    while(cond)
        for j = 1:dim_y
            if(i == j)
                p = p_s;
            else
                p = p_ns;
            end
            if(rand >= p)
                A(i,j) = 0;
            else
                A(i,j) = 1;
            end
        end
        if(sum(A(i, :))~=0)
            cond = false;
        end
    end
end
C = C.*A;

% Generate the data
y(:, 1) = rand(dim_y, 1);
for t = 2:T
    y(:, t) = C*y(:, t-1) + mvnrnd(zeros(1, dim_y), var_u*eye(dim_y))';
end

end
