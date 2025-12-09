
function [clusterLabels,t,gap_lm] = alg6Rw(W, D, F, k)
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
I = speye(n);                         

% Compute G varient
G = D \ W + 2 * I;   
% M = F'*G;
M = F' * W;
% U = G - G*F*((M*F)\M);


tic;
[H, vals] = eigs(@(b) SMW_Afun(b,M,F,G) , n , k, 'lr','MaxIterations',5000);
% [H,vals] = eigs(U, k, 'lr', 'MaxIterations',1000,'SubspaceDimension',4*k);
t = toc;
gap_lm = 0;

% eigvals_lm = sort(diag(vals),'descend');
% gap_lm = eigvals_lm(k) - eigvals_lm(k+1);
% fprintf('  ALGRW LM gap = %.4e\n', gap_lm);

clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);