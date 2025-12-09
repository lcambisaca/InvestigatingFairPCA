function [clusterLabels,t] = alg1(W, D, k)
%INPUT:
%   W ... (weighted) adjacency matrix of size n x n
%   D ... degree matrix of W
%   F ... group membership matrix G of size n x (h-1)
%   k ... number of clusters
%
%OUTPUT:
% clusterLabels ... vector of length n comprising the cluster label for each
%                  data point
% t ... CPU time of eigs
%-----------------------------------------------------------------------------%
    n = size(W, 1);

    % Make Sparse
    W = sparse(W); 
    D = spdiags(diag(D), 0, n, n);   
    sqrtD = spdiags(sqrt(diag(D)), 0, n, n);

    %Normal Code
    L = D - W;
    Ln = (sqrtD\L)/sqrtD;
    % Ln = (Ln+Ln')/2;
   
    tic
    [Y, vals] = eigs(@(b) Afun2(Ln, b), n, k, 'sr', 'MaxIterations',5000, 'SubspaceDimension',4*k);
    t = toc;
    H = sqrtD\Y;
    clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
    
   

end