function P_star = fair_pca(A, B, d)
% Implements the Fair PCA algorithm using CVX.

% INPUTS:
%   A - m1 x n matrix for group A
%   B - m2 x n matrix for group B
%   d - The target "soft" dimension for the projection (from Eq. 8)
%
% OUTPUT:
%   U_fair - (m1+m2) x n matrix containing the transformed data


    % --- Step 1: Get dimensions and set up data ---
    % We will use the original A and B as A_hat and B_hat
    [m1, n] = size(A);
    [m2, ~] = size(B);
    A_hat = A;
    B_hat = B;

    % --- Step 2: Solve the "Fair" SDP using CVX ---
    cvx_begin sdp quiet
        % 1. Define CVX variables
        % P_hat is the (n x n) symmetric matrix we are solving for
        variable P_hat(n, n) symmetric 
        variable z(1)
        
        % 2. Pre-compute constant terms
        C_A = A_hat' * A_hat;
        C_B = B_hat' * B_hat;
        norm_A_sq = norm(A_hat, 'fro')^2;
        norm_B_sq = norm(B_hat, 'fro')^2;
        
        % 3. Define the error expressions (from Eq. 6 & 7)
        error_A = (1/m1) * (norm_A_sq - trace(C_A * P_hat));
        error_B = (1/m2) * (norm_B_sq - trace(C_B * P_hat));
        
        % 4. Define and solve the optimization problem
        minimize(z)
        subject to
            z >= error_A;          % Eq. 6
            z >= error_B;          % Eq. 7
            trace(P_hat) <= d;     % Eq. 8
            P_hat >= 0;            % Eq. 9 (P_hat is positive semidefinite)
            P_hat <= eye(n);       % Eq. 9 (I - P_hat is also positive semidefinite)
    cvx_end
    
    % --- Step 3: Analyze the Solution (Eigendecomposition) ---
    % P_hat is now the numerical solution.
    % For a symmetric matrix, eigendecomposition is the same as SVD.
    % [U, D] = eig(X) -> X = U*D*U'
    % U_svd contains the eigenvectors (u_j)
    % D contains the eigenvalues (lambda_bar_j)
    [U_svd, D] = eig(P_hat);
    lambda_bar_j = diag(D); % Get eigenvalues from diagonal matrix
    

    % --- Step 5: Round the Solution ---
    
    % Clip eigenvalues to [0, 1] to avoid numerical errors (e.g., sqrt(-1e-15))
    lambda_bar_clipped = max(0, min(1, lambda_bar_j));
    
    % Apply the rounding formula from the paper
    lambda_star_j = 1 - sqrt(1 - lambda_bar_clipped);
    
    % Rebuild the final matrix P*
    % P* = U * diag(lambda*) * U'
    P_star = U_svd * diag(lambda_star_j) * U_svd';    
    
end