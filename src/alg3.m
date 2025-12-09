function [clusterLabels,t,gap_sm] = alg3(W, D, F, k) % S-Fair-SC
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
C = sqrtD\F;
Ln = (sqrtD\L)/sqrtD;

% Ln = (Ln+Ln')/2;

sigma = norm(Ln,1);


tic
[X, vals] = eigs(@(b) Afun(Ln, C, b, sigma), n, k, 'sr','MaxIterations', 5000);
t = toc;
gap_sm = 0;

% eigvals_sm = sort(diag(vals),'ascend');
% gap_sm = eigvals_sm(k+1) - eigvals_sm(k);
% fprintf('  SM gap = %.4e\n', gap_sm);

% disp("Alg3")
% % Plot
% figure;
% plot(1:length(eigvals_sm), eigvals_sm, 'bo-', 'LineWidth', 2);
% hold on;
% xline(k, '--r', 'k cutoff', 'LineWidth', 1.5);
% title(['S-Fair-SC: Absolute Eigengap']);
% xlabel('Eigenvalue Index (sorted)');
% ylabel('Eigenvalue Magnitude');
% grid on;
% 
% % Compute eigengap
% text(k + 0.5, mean([eigvals_sm(k), eigvals_sm(k+1)]), ...
%     ['Eigengap = ', num2str(gap_sm, '%.4f')], ...
%     'Color', 'blue', 'FontSize', 12);
% pause


H = sqrtD\X;
clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);
end