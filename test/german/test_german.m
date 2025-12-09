clc; clear; close all;
addpath('../')
color = readmatrix('german_colors.csv');
color = color(:,2) + 1;
m = max(color);

edges = readmatrix('german_edges.csv');
EdgeTable = table(edges+1,'VariableNames',{'EndNodes'});
G = graph(EdgeTable);


W = G.adjacency;
n = size(W, 1);
sensitive = color;
% converting sensitive to a vector with entries in [h] and building F %%%
sens_unique=unique(sensitive);
h = length(sens_unique);
sens_unique=reshape(sens_unique,[1,h]);
sensitiveNEW=sensitive;
temp=1;
for ell=sens_unique
sensitiveNEW(sensitive==ell)=temp;
temp=temp+1;
end
F=zeros(n,h-1);
for ell=1:(h-1)
temp=(sensitiveNEW == ell);
F(temp,ell)=1;
groupSize = sum(temp);
F(:,ell) = F(:,ell)-groupSize/n;
end
degrees = sum(W, 1);
D = diag(degrees);


% fair = false;
% [D,W] = SCPca(W, D, (sensitiveNEW == 1), G, F, fair);
% Clustering range and number of runs
krange = 2:1:15;
numRuns = 1;

time_alg1 = zeros(length(krange), numRuns); %Nomalized-SC
% time_alg2 = zeros(length(krange), numRuns); %Fair-SC
time_alg3 = zeros(length(krange), numRuns); %S-Fair-SC
% time_alg4Rw = zeros(length(krange), numRuns); %RW-SVM-OLD:  G = D\W
% time_alg4Sym = zeros(length(krange), numRuns); %SYM-SVM-OLD: G = (sqrtD\W)/sqrtD
time_alg5 = zeros(length(krange), numRuns); %W+n
time_alg6Rw = zeros(length(krange), numRuns); %RW-SVM-SC: G = D\W + n * eye(n)
time_alg6Sym = zeros(length(krange), numRuns); %SYM-SVM-SC: G = (sqrtD\W)/sqrtD +  n * eye(n)

% To store balance values across runs
balance1_all = zeros(length(krange), numRuns);
% balance2_all = zeros(length(krange), numRuns);
balance3_all = zeros(length(krange), numRuns);
% balance4Rw_all = zeros(length(krange), numRuns);
% balance4Sym_all = zeros(length(krange), numRuns);
balance5_all = zeros(length(krange), numRuns);
balance6Rw_all = zeros(length(krange), numRuns);
balance6Sym_all = zeros(length(krange), numRuns);

eigs_alg1 = zeros(length(krange), numRuns);
% eigs_alg2 = zeros(length(krange), numRuns);
eigs_alg3 = zeros(length(krange), numRuns);
% eigs_alg4Rw = zeros(length(krange), numRuns);
% eigs_alg4Sym = zeros(length(krange), numRuns);
eigs_alg5 = zeros(length(krange), numRuns);
eigs_alg6Rw = zeros(length(krange), numRuns);
eigs_alg6Sym = zeros(length(krange), numRuns);


for run = 1:numRuns
    fprintf('Run %d/%d\n', run, numRuns);
    for i = 1:length(krange)
        k = krange(i);
        fprintf('K: %d/%d\n', i+1, length(krange) + 1);
 
        tstart1 = tic;
        [labelsalg1 , t1] = alg1(W, D, k);
        time_alg1(i,run) = toc(tstart1);
        eigs_alg1(i,run) = t1;
        balance1_all(i, run) = computeBalanceg(labelsalg1, sensitive==1, k);
        
        % tstart2 = tic;
        % [labelsalg2 , t2] = alg2(W, D, F, k);
        % time_alg2(i,run) = toc(tstart2);
        % eigs_alg2(i,run) = t2;
        % balance2_all(i, run) = computeBalanceg(labelsalg2, sensitive==1, k);

        tstart3 = tic;
        [labelsalg3 , t3] = alg3(W, D, F, k);
        time_alg3(i,run) = toc(tstart3);
        eigs_alg3(i,run) = t3;
        balance3_all(i, run) = computeBalanceg(labelsalg3, sensitive==1, k);


        % tstart4Rw = tic;
        % [labelsalg4Rw , t4] = alg4Rw(W, D, F, k); 
        % time_alg4Rw(i,run) = toc(tstart4Rw);
        % eigs_alg4Rw(i,run) = t4;
        % balance4Rw_all(i, run) = computeBalanceg(labelsalg4Rw, sensitive==1, k);

        % tstart4Sym = tic;
        % [labelsalg4Sym , t5] = alg4Sym(W, D, F, k);
        % eigs_alg4Sym(i,run) = t5;
        % time_alg4Sym(i,run) = toc(tstart4Sym);
        % balance4Sym_all(i, run) = computeBalanceg(labelsalg4Sym, sensitive==1, k);

        tstart5 = tic;
        [labelsalg5, t6] = alg5(W, D, F, k);
        time_alg5(i, run) = toc(tstart5);
        eigs_alg5(i, run) = t6;
        balance5_all(i, run) = computeBalanceg(labelsalg5, sensitive==1, k);

        
        tstart6Rw = tic;
        [labels6Rw, t7] = alg6Rw(W, D, F, k);
        time_alg6Rw(i, run) = toc(tstart6Rw);
        eigs_alg6Rw(i, run) = t7;
        balance6Rw_all(i, run) = computeBalanceg(labels6Rw, sensitive==1, k);

        tstart6Sym = tic;
        [labels6Sym, t8] = alg6Sym(W, D, F, k);
        time_alg6Sym(i, run) = toc(tstart6Sym);
        eigs_alg6Sym(i, run) = t8;
        balance6Sym_all(i, run) = computeBalanceg(labels6Sym, sensitive==1, k);

    end
end


% Compute mean time
mean_time1 = mean(time_alg1, 2);
% mean_time2 = mean(time_alg2, 2);
mean_time3 = mean(time_alg3, 2);
% mean_time4Rw = mean(time_alg4Rw, 2);
% mean_time4Sym = mean(time_alg5Sym, 2);
mean_time5 = mean(time_alg5, 2);
mean_time6Rw = mean(time_alg6Rw, 2);
mean_time6Sym = mean(time_alg6Sym, 2);


% Compute mean eigs time
mean_eigs_time1 = mean(eigs_alg1, 2);
% mean_eigs_time2 = mean(eigs_alg2, 2);
mean_eigs_time3 = mean(eigs_alg3, 2);
% mean_eigs_time4Rw = mean(eigs_alg4Rw, 2);
% mean_eigs_time4Sym = mean(eigs_alg5Sym, 2);
mean_eigs_time5 = mean(eigs_alg5, 2);
mean_eigs_time6Rw = mean(eigs_alg6Rw, 2);
mean_eigs_time6Sym = mean(eigs_alg6Sym, 2);

% Compute mean balance
mean_balance1 = mean(balance1_all, 2);
% mean_balance2 = mean(balance2_all, 2);
mean_balance3 = mean(balance3_all, 2);
% mean_balance4Rw = mean(balance4Rw_all, 2);
% mean_balance4Sym = mean(balance4Sym_all, 2);
mean_balance5 = mean(balance5_all, 2);
mean_balance6Rw = mean(balance6Rw_all, 2);
mean_balance6Sym = mean(balance6Sym_all, 2);



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
% plot(krange, mean_balance2, 'mo-');
plot(krange, mean_balance3, 'rd-');
% plot(krange, mean_balance4Rw, 'r^-');
% plot(krange, mean_balance4Sym, 'k+-');
plot(krange, mean_balance5, 'gs-');
plot(krange, mean_balance6Sym, 'c*-');
plot(krange, mean_balance6Rw, 'mo-');

legend({'SC','S-Fair-SC','AFF-Fair-SMW','SYM-Fair-SNW','RW-Fair-SMW'}, 'Location','northwest', 'FontSize',9)
xlabel('k (Clusters)');
ylabel('Mean Balance');
title(sprintf('Mean Balance Metric over %d runs (German Data Set)', numRuns), 'FontWeight', 'normal');
grid on;
hold off


figure;
clf;
hold on
  plot(krange, mean_time1, 'bx-');
% plot(krange, mean_time2, 'mo-');
  plot(krange, mean_time3, 'rd-');
% plot(krange, mean_time4Rw, 'r^-');
% plot(krange, mean_time4Sym, 'k+-');
  plot(krange, mean_time5, 'gs-');
  plot(krange, mean_time6Sym, 'c*-');
  plot(krange, mean_time6Rw, 'mo-');

legend({'SC','S-Fair-SC','AFF-Fair-SMW','SYM-Fair-SNW','RW-Fair-SMW'}, 'Location','northwest', 'FontSize',9)
xlabel('k (Clusters)')
ylabel('Running time (s)')
title(sprintf('Algorithms Run Time over %d runs (German Data Set)', numRuns), 'FontWeight', 'normal');
grid on
hold off

figure;
clf;
hold on
plot(krange, mean_eigs_time1, 'bx-');
% plot(krange, mean_eigs_time2, 'mo-');
plot(krange, mean_eigs_time3, 'rd-');
% plot(krange, mean_eigs_time4Rw, 'r^-');
% plot(krange, mean_eigs_time4Sym, 'k+-');
plot(krange, mean_eigs_time5, 'gs-');
plot(krange, mean_eigs_time6Sym, 'c*-');
plot(krange, mean_eigs_time6Rw, 'mo-');

legend({'SC','S-Fair-SC','AFF-Fair-SMW','SYM-Fair-SNW','RW-Fair-SMW'}, 'Location','northwest', 'FontSize',9)
xlabel('k (Clusters)')
ylabel('Running time (s)')
title(sprintf('Algorithms Eigs Run Time over %d runs (German Data Set)', numRuns), 'FontWeight', 'normal');
grid on
hold off
        