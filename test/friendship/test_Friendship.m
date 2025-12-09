clc; clear; close all;
addpath('../')

group = readmatrix('group_clean.csv');
friendship = readmatrix('friendship_clean.csv');

nall = size(group, 1);
A = zeros(nall, nall);
for m = 1:size(friendship,1)
    i = find(group(:,1) == friendship(m,1));
    j = find(group(:,1) == friendship(m,2));
    if friendship(m,3) == 1
        A(i,j) = 1;
        A(j,i) = 1;
    end
end

G = graph(A);

% find the largest connected component of graph G
[bin,binsize] = conncomp(G);
idx = binsize(bin) == max(binsize);
SG = subgraph(G, idx);

% W matrix
W = adjacency(SG);
n = size(W, 1);

% D matrix
D = diag(W*ones(n,1));

% F matrix
g = group(idx,:);
gmale = g(:,2);
gfemale = double(~gmale);
F = gfemale - sum(gfemale)/n;

fair = true;
[D,W] = SCPca(W, D, gmale, SG, F, fair);

pause
% Clustering range and number of runs
krange = 2:1:20;
numRuns = 1;

time_alg1 = zeros(length(krange), numRuns); %Nomalized-SC
time_alg3 = zeros(length(krange), numRuns); %S-Fair-SC
time_alg5 = zeros(length(krange), numRuns); %W+n

% To store balance values across runs
balance1_all = zeros(length(krange), numRuns);
balance3_all = zeros(length(krange), numRuns);
balance5_all = zeros(length(krange), numRuns);

eigs_alg1 = zeros(length(krange), numRuns);
eigs_alg3 = zeros(length(krange), numRuns);
eigs_alg5 = zeros(length(krange), numRuns);

for run = 1:numRuns
    fprintf('Run %d/%d\n', run, numRuns);
    for i = 1:length(krange)
        k = krange(i);
        fprintf('K: %d/%d\n', i+1, length(krange) + 1);

        
        tstart1 = tic;
        [labelsalg1 , t1] = alg1(W, D, k);
        time_alg1(i,run) = toc(tstart1);
        eigs_alg1(i,run) = t1;
        balance1_all(i, run) = computeBalance(labelsalg1, gmale, k);
        

        tstart3 = tic;
        [labelsalg3 , t3] = alg3(W, D, F, k);
        time_alg3(i,run) = toc(tstart3);
        eigs_alg3(i,run) = t3;
        balance3_all(i, run) = computeBalance(labelsalg3, gmale, k);

        tstart5 = tic;
        [labelsalg5, t6] = alg5(W, D, F, k);
        time_alg5(i, run) = toc(tstart5);
        eigs_alg5(i, run) = t6;
        balance5_all(i, run) = computeBalance(labelsalg5, gmale, k);


    end
end


% Compute mean time
mean_time1 = mean(time_alg1, 2);
mean_time3 = mean(time_alg3, 2);
mean_time5 = mean(time_alg5, 2);



% Compute mean eigs time
mean_eigs_time1 = mean(eigs_alg1, 2);
mean_eigs_time3 = mean(eigs_alg3, 2);
mean_eigs_time5 = mean(eigs_alg5, 2);


% Compute mean balance
mean_balance1 = mean(balance1_all, 2);
mean_balance3 = mean(balance3_all, 2);
mean_balance5 = mean(balance5_all, 2);


% set default sizes for figures:
ulesfontsize = 15;
set(0, 'DefaultAxesFontSize', ulesfontsize);
set(0, 'DefaultTextFontSize', ulesfontsize);
set(0, 'DefaultUIControlFontSize', ulesfontsize);
set(0,'DefaultLineMarkerSize',ulesfontsize);
set(0,'DefaultLineLineWidth',2.5) 

figure;
clf;
hold on
plot(krange, mean_balance1, 'bx-');
plot(krange, mean_balance3, 'rd-');
plot(krange, mean_balance5, 'gs-');


legend({'SC','S-Fair-SC','AFF-Fair-SMW'}, 'Location','northwest', 'FontSize',9)
xlabel('k (Clusters)');
ylabel('Mean Balance');
title(sprintf('Mean Balance Metric over %d runs (Friendship Data Set)', numRuns), 'FontWeight', 'normal');
grid on;
hold off

% 
% figure;
% clf;
% hold on
%   plot(krange, mean_time1, 'bx-');
%   plot(krange, mean_time3, 'rd-');
%   plot(krange, mean_time5, 'gs-');
% 
% legend({'SC','S-Fair-SC','AFF-Fair-SMW'}, 'Location','northwest', 'FontSize',9)
% xlabel('k (Clusters)')
% ylabel('Running time (s)')
% title(sprintf('Algorithms Run Time over %d runs (Friendship Data Set)', numRuns), 'FontWeight', 'normal');
% grid on
% hold off
% 
% figure;
% clf;
% hold on
% plot(krange, mean_eigs_time1, 'bx-');
% plot(krange, mean_eigs_time3, 'rd-');
% plot(krange, mean_eigs_time5, 'gs-');
% 
% 
% legend({'SC','S-Fair-SC','AFF-Fair-SMW'}, 'Location','northwest', 'FontSize',9)
% xlabel('k (Clusters)')
% ylabel('Running time (s)')
% title(sprintf('Algorithms Eigs Run Time over %d runs (Friendship Data Set)', numRuns), 'FontWeight', 'normal');
% grid on
% hold off
