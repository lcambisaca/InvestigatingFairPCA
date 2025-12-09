
function [clusterLabels, t] = alg2(W, D, F, k) % Fair-SC
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
L = D - W;
Z = null(F');

Q=sqrtm(Z'*D*Z);

M=(Q\Z')*L*(Z/Q); 
M=(M+M')/2;

tic
[Y, vals] = eigs(@(b) Afun2(M, b), n - size(F,2), k,'sr','MaxIterations',1000,'SubspaceDimension',4*k);
t = toc;

H = Z*(Q\Y);
clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);

end

