
function [clusterLabels ,t,gap_lm] = alg5(W, D, F, k) 
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
I = speye(n);

% Compute G varient
G = W +  I * n;
% G = (G+G')/2;

M = F'*G;
% U = G - G*F*((M*F)\M);

tic;
[H, vals] = eigs(@(b) SMW_Afun(b,M,F,G) , n , k, 'lr','MaxIterations',5000);
t = toc;
gap_lm = 0;
% eigvals_lm = sort(diag(vals),'descend');
% gap_lm = eigvals_lm(k) - eigvals_lm(k+1);
% fprintf('  LM gap = %.4e\n', gap_lm);

% disp("Alg5")
% % Plot
% figure;
% plot(1:length(eigvals_lm), eigvals_lm, 'bo-', 'LineWidth', 2);
% hold on;
% xline(k, '--r', 'k', 'LineWidth', 1.5);
% title(['AFF-SMW-SC: Absolute Eigengap']);
% xlabel('Eigenvalue Index (sorted)');
% ylabel('Eigenvalue Magnitude');
% grid on;
% 
% % Compute eigengap
% text(k + 0.5, mean([eigvals_lm(k), eigvals_lm(k+1)]), ...
%     ['Eigengap = ', num2str(gap_lm, '%.4f')], ...
%     'Color', 'blue', 'FontSize', 12);
% pause

clusterLabels = kmeans(H,k,'Replicates',10, 'MaxIter',500);





