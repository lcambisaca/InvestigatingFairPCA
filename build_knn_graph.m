function [W] = build_knn_graph(Z, k_nn)
% Builds a k-Nearest Neighbor (k-NN) graph from a feature matrix.
% The graph is unweighted and symmetric.
%
% INPUTS:
%   Z:      [n, p] feature matrix (n=samples, p=features)
%   k_nn:   Number of neighbors to connect.
%
% OUTPUT:
%   W:      [n, n] sparse adjacency matrix

    n = size(Z, 1);

    [idx] = knnsearch(Z, Z, 'K', k_nn + 1);
    
    neighbors = idx(:, 2:end); % [n, k_nn]
    
    row_indices = repelem((1:n)', k_nn, 1);
    
    col_indices = reshape(neighbors, [], 1);
    
    % Create a sparse matrix with 1s at these (i, j) locations
    W_sparse = sparse(row_indices, col_indices, 1, n, n);
    
    W = W_sparse + W_sparse';

    W = (W > 0);
    W = double(W); % Convert from logical back to double
end