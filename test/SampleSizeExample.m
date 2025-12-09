clear; clc; close all;
rng(42);

% ---------------------------------------------------------------
%  PARAMETERS
% ---------------------------------------------------------------

n = 20;           % number of features
featureNum = 2;   % PCA/Fair PCA projected dimensions
n_A1 = 15000;       % male samples
n_B1 = 1000;       % female samples

% ---------------------------------------------------------------
%  GENERATE DATA — SCENARIO 3
%  Male = high SampleSize, Female = low SampleSize
% ---------------------------------------------------------------

mu_A1 = zeros(1, n);
mu_B1 = zeros(1, n);


sigma_A1 = diag(ones(n,1)); 
sigma_A1(1,1) = 40;  % High Variance
sigma_A1(2,2) = 40;  % High Variance
sigma_A1(3,3) = 1;   % Low Variance
sigma_A1(4,4) = 1;   % Low Variance

sigma_B1 = diag(ones(n,1));
sigma_B1(1,1) = 1;   % Low Variance
sigma_B1(2,2) = 1;   % Low Variance
sigma_B1(3,3) = 40;  % High Variance
sigma_B1(4,4) = 40;  % High Variance

A = mvnrnd(mu_A1, sigma_A1, n_A1);   % Male
B = mvnrnd(mu_B1, sigma_B1, n_B1);   % Female

M = [A; B];                          % Full 

Group = [zeros(n_A1,1); ones(n_B1,1)]; % 0 = Male, 1 = Female

% ---------------------------------------------------------------
%  REAL CREDITWORTHINESS LABELS
%  Only first 3 dims matter
% ---------------------------------------------------------------

w1 = [1,1,zeros(1,n-2)];
w2 = [0,0,1,1,zeros(1,n-4)];

epsA = randn(n_A1,1) * 5;
epsB = randn(n_B1,1) * 5;

scoreA = A * w1' + epsA;
scoreB = B * w2' + epsB;

% Balanced threshold
th = median([scoreA; scoreB]);

Y_A = (scoreA > th);
Y_B = (scoreB > th);

Y = [Y_A; Y_B];

%% ---------------------------------------------------------------
%  RUN PCA AND FAIR PCA
% ---------------------------------------------------------------

coeff = pca(M);
loss_A = zeros(featureNum,1);
loss_B = zeros(featureNum,1);
lossFair_A = zeros(featureNum,1);
lossFair_B = zeros(featureNum,1);
lossFair_max = zeros(featureNum,1);
lossHarshFair_A = zeros(featureNum,1);
lossHarshFair_B = zeros(featureNum,1);
lossHarshFair_max = zeros(featureNum,1);
tcMWPCA = zeros(featureNum,1);
tcSDPPCA = zeros(featureNum,1);

eta = 1;
T = 10;

z = zeros(featureNum,1);
z_last = zeros(featureNum,1);

for ell = 1:featureNum

    % PCA projection matrix
    P = coeff(:,1:ell) * coeff(:,1:ell)';

    % Reconstruction
    approx_A = A * P;
    approx_B = B * P;

    % PCA reconstruction loss
    loss_A(ell) = loss(A, approx_A, ell) / size(A,1);
    loss_B(ell) = loss(B, approx_B, ell) / size(B,1);

    % FAIR PCA (via MW algorithm)
    tic
    [P_fair, z(ell), P_last, z_last(ell)] = mw(A, B, ell, eta, T);
    tcMWPCA(ell) = toc;
    
    

    % Smart selection
    if z(ell) < z_last(ell)
        P_smart = P_fair;
    else
        P_smart = P_last;
    end

    % Round to projection
    P_smart = eye(size(P_smart,1)) - sqrtm(eye(size(P_smart,1)) - P_smart);
    tic
    harshFair = fair_pca(A, B, ell);
    tcSDPPCA(ell) = toc;
    

    % Fair reconstruction
    approxFair_A = A * P_smart;
    approxFair_B = B * P_smart;

    % Harsh reconstruction
    approxHarshFair_A = A * harshFair;
    approxHarshFair_B = B * harshFair;

    % Fair reconstruction loss
    lossFair_A(ell) = loss(A, approxFair_A, ell)/size(A,1);
    lossFair_B(ell) = loss(B, approxFair_B, ell)/size(B,1);
    lossFair_max(ell) = max([lossFair_A(ell), lossFair_B(ell)]);

    % Harsh Fair reconstruction loss
    lossHarshFair_A(ell) = loss(A, approxHarshFair_A, ell)/size(A,1);
    lossHarshFair_B(ell) = loss(B, approxHarshFair_B, ell)/size(B,1);
    lossHarshFair_max(ell) = max([lossHarshFair_A(ell), lossHarshFair_B(ell)]);

end

% ---------------------------------------------------------------
%  PLOT PCA VS FAIR PCA LOSS
% ---------------------------------------------------------------

figure; hold on;
plot(1:featureNum, loss_A,'gx-','LineWidth',2);
plot(1:featureNum, loss_B,'ro-','LineWidth',2);
plot(1:featureNum, lossHarshFair_max,'y--*','LineWidth',2);
plot(1:featureNum, lossFair_max,'b--o','LineWidth',2);


title('PCA vs Fair-PCA Reconstruction Loss');
legend('Male (PCA)','Female (PCA)','SDP Fair PCA','MW Fair PCA');
xlabel('Projected Dimensions');
ylabel('Average Reconstruction Loss');
grid on;

figure; hold on;
plot(1:featureNum, tcSDPPCA,'gx-','LineWidth',2);
plot(1:featureNum, tcMWPCA,'ro-','LineWidth',2);

title('PCA vs Fair-PCA Time Complexity');
legend('SDP Fair PCA','MW Fair PCA');
xlabel('Projected Dimensions');
ylabel('Average Reconstruction Loss');
grid on;


%% ---------------------------------------------------------------
%  CREATE PCA AND FAIR PCA FEATURE SPACES
% ---------------------------------------------------------------

% P_PCA = coeff(:,1:featureNum) * coeff(:,1:featureNum)';
% M_PCA = M * P_PCA;

% 1. PCA Projection Matrix

[coeff, score_all, ~] = pca(M);
M_PCA = score_all(:, 1:featureNum); % This is N x 3


% 2. FPCA Projection Matrix

P_smart = (P_smart + P_smart') / 2;
[V_fair, D_fair] = eig(P_smart);
[d_diag, ind] = sort(diag(D_fair), 'descend');
V_fair_sorted = V_fair(:, ind);
basis_fair = V_fair_sorted(:, 1:featureNum);
M_FairPCA = real(M * basis_fair);

% 3. SDP Counterpart SuperHarsh
harshFair = fair_pca(A, B, featureNum);
harshFair = (harshFair + harshFair')/2;
[V_fair, D_fair] = eig(harshFair);
[d_diag, ind] = sort(diag(D_fair), 'descend');
V_fair_sorted = V_fair(:, ind);
basis_harsh_fair = V_fair_sorted(:, 1:featureNum);
M_HarshFairPCA = real(M * harshFair);



%% ---------------------------------------------------------------
%  VISUALIZATION: WHICH FEATURES DID THEY PICK?
% ---------------------------------------------------------------
% 1. Calculate "Importance" (Sum of absolute weights across the k dims)
% We use absolute value because -0.9 is just as important as 0.9
importance_pca = sum(abs(coeff(:, 1:featureNum)), 2);
importance_fair = sum(abs(basis_fair), 2);
importance_harsh_fair = sum(abs(basis_harsh_fair), 2);


% Normalize them so they are easy to compare on one plot
importance_pca = importance_pca / max(importance_pca);
importance_fair = importance_fair / max(importance_fair);
importance_harsh_fair = importance_harsh_fair / max(importance_harsh_fair); 
% 2. Plot Side-by-Side
figure;
b = bar(1:n, [importance_pca, importance_fair,importance_harsh_fair]);
b(1).FaceColor = 'r'; % Red for Standard PCA
b(2).FaceColor = 'b'; % Blue for Fair PCA
b(3).FaceColor = 'g'; % Blue for Fair PCA


% Formatting
title('Feature Importance: Which Columns were Picked?');
xlabel('Original Feature Index');
ylabel('Relative Importance (Normalized Weight)');
legend('Standard PCA', 'MW Fair PCA','SDP Fair PCA');
grid on;

% Highlight the critical features
xticks(1:10); % Just show first 10 for clarity
xlim([0, 6]); % Zoom in on the important ones
%% ---------------------------------------------------------------
%  TRAIN/TEST SPLIT
% ---------------------------------------------------------------

cv = cvpartition(Y,'HoldOut',0.2);
idxTrain = training(cv);
idxTest  = test(cv);

Xtrain_pca = M_PCA(idxTrain,:);
Xtest_pca  = M_PCA(idxTest,:);

Xtrain_fpca = M_FairPCA(idxTrain,:);
Xtest_fpca  = M_FairPCA(idxTest,:);


Xtrain_harshfpca = M_HarshFairPCA(idxTrain,:);
Xtest_harshfpca  = M_HarshFairPCA(idxTest,:);

ytrain = Y(idxTrain);
ytest  = Y(idxTest);

Group_test = Group(idxTest);

%% ---------------------------------------------------------------
%  CLASSIFIERS (SVM Implementation)
% ---------------------------------------------------------------
% 1. Train SVM on Standard PCA Data
mdl_pca = fitcsvm(Xtrain_pca, ytrain, 'KernelFunction', 'linear', 'Standardize', true);
% 2. Train SVM on Fair PCA Data
mdl_fpca = fitcsvm(Xtrain_fpca, ytrain, 'KernelFunction', 'linear', 'Standardize', true);
% 3. Train SVM on SDP (Harsh) Fair PCA Data
mdl_harshfpca = fitcsvm(Xtrain_harshfpca, ytrain, 'KernelFunction', 'linear', 'Standardize', true);


[yhat_pca, scores_pca] = predict(mdl_pca, Xtest_pca);
[yhat_fpca, scores_fpca] = predict(mdl_fpca, Xtest_fpca);
[yhat_harshfpca, scores_harshfpca] = predict(mdl_harshfpca, Xtest_harshfpca);

%% ---------------------------------------------------------------
%  METRICS: BALANCED ACCURACY (Addressing the 15x Magnitude Issue)
% ---------------------------------------------------------------
male_mask   = (Group_test == 0);
female_mask = (Group_test == 1);

% 1. Calculate Accuracy specifically for each group
acc_pca_male   = mean(yhat_pca(male_mask)   == ytest(male_mask));
acc_pca_female = mean(yhat_pca(female_mask) == ytest(female_mask));
bal_acc_pca    = (acc_pca_male + acc_pca_female) / 2;

acc_fpca_male   = mean(yhat_fpca(male_mask)   == ytest(male_mask));
acc_fpca_female = mean(yhat_fpca(female_mask) == ytest(female_mask));
bal_acc_fpca    = (acc_fpca_male + acc_fpca_female) / 2;

acc_harshfpca_male   = mean(yhat_harshfpca(male_mask)   == ytest(male_mask));
acc_harshfpca_female = mean(yhat_harshfpca(female_mask) == ytest(female_mask));
bal_acc_harshfpca    = (acc_harshfpca_male + acc_harshfpca_female) / 2;

fprintf('--------------------------------------------------\n');
fprintf('GLOBAL VS BALANCED ACCURACY\n');
fprintf('--------------------------------------------------\n');
fprintf('PCA Global Accuracy:      %.3f\n', mean(yhat_pca == ytest));
fprintf('PCA Balanced Accuracy:    %.3f  <-- Real Performance\n', bal_acc_pca);
fprintf('PCA Gap (Male - Fem):     %.3f\n', acc_pca_male - acc_pca_female);
fprintf('\n');
fprintf('MWFair-PCA Global Accuracy:   %.3f\n', mean(yhat_fpca == ytest));
fprintf('MWFair-PCA Balanced Accuracy: %.3f  <-- Real Performance\n', bal_acc_fpca);
fprintf('MWFair-PCA Gap (Male - Fem):  %.3f\n', acc_fpca_male - acc_fpca_female);
fprintf('\n');
fprintf('SDPFair-PCA Global Accuracy:   %.3f\n', mean(yhat_harshfpca == ytest));
fprintf('SDPFair-PCA Balanced Accuracy: %.3f  <-- Real Performance\n', bal_acc_harshfpca);
fprintf('SDPFair-PCA Gap (Male - Fem):  %.3f\n', acc_harshfpca_male - acc_harshfpca_female);

%% ---------------------------------------------------------------
%  GROUP-FAIRNESS METRICS (Corrected Definitions)
% ---------------------------------------------------------------
% Helper functions for True Positive Rate (Recall) and False Positive Rate
calc_tpr = @(y, yhat) sum(y==1 & yhat==1) / sum(y==1);
calc_fpr = @(y, yhat) sum(y==0 & yhat==1) / sum(y==0);

% PCA Metrics
TPR_pca_m = calc_tpr(ytest(male_mask), yhat_pca(male_mask));
TPR_pca_f = calc_tpr(ytest(female_mask), yhat_pca(female_mask));
FPR_pca_m = calc_fpr(ytest(male_mask), yhat_pca(male_mask));
FPR_pca_f = calc_fpr(ytest(female_mask), yhat_pca(female_mask));

% Fair PCA Metrics
TPR_fpca_m = calc_tpr(ytest(male_mask), yhat_fpca(male_mask));
TPR_fpca_f = calc_tpr(ytest(female_mask), yhat_fpca(female_mask));
FPR_fpca_m = calc_fpr(ytest(male_mask), yhat_fpca(male_mask));
FPR_fpca_f = calc_fpr(ytest(female_mask), yhat_fpca(female_mask));

% Fair SDPPCA Metrics
TPR_harshfpca_m = calc_tpr(ytest(male_mask), yhat_harshfpca(male_mask));
TPR_harshfpca_f = calc_tpr(ytest(female_mask), yhat_harshfpca(female_mask));
FPR_harshfpca_m = calc_fpr(ytest(male_mask), yhat_harshfpca(male_mask));
FPR_harshfpca_f = calc_fpr(ytest(female_mask), yhat_harshfpca(female_mask));

fprintf('\n--------------------------------------------------\n');
fprintf('FAIRNESS METRICS (Opportunity & Odds)\n');
fprintf('--------------------------------------------------\n');
fprintf('             | Male TPR | Female TPR |  GAP  |\n');
fprintf('Standard PCA |  %.3f   |   %.3f    | %.3f |\n', TPR_pca_m, TPR_pca_f, TPR_pca_m - TPR_pca_f);
fprintf('Fair PCA     |  %.3f   |   %.3f    | %.3f |\n', TPR_fpca_m, TPR_fpca_f, TPR_fpca_m - TPR_fpca_f);
fprintf('SDPFair PCA  |  %.3f   |   %.3f    | %.3f |\n', TPR_harshfpca_m, TPR_harshfpca_f, TPR_harshfpca_m - TPR_harshfpca_f);


%%

k_nn = 5; % ----- FAIR PCA -----

W_PCA = build_knn_graph(M_PCA, k_nn);
D_PCA = diag(sum(W_PCA,2));

W_Fair = build_knn_graph(M_FairPCA, k_nn);
D_Fair = diag(sum(W_Fair,2));

W_norm = build_knn_graph(M, k_nn);
D_norm = diag(sum(W_norm,2));

W_harsh = build_knn_graph(M_HarshFairPCA, k_nn);
D_harsh = diag(sum(W_harsh,2));

gfemale = [zeros(n_A1,1); ones(n_B1,1)];
gmale = 1 - gfemale;
F = gfemale - mean(gfemale);    % Centered sensitive vector

% ---- 6. Run Clustering ----

[PCA_alg1, PCA_alg3, PCA_alg5] = runClustering(W_PCA, D_PCA, F, gmale);
[FPCA_alg1, FPCA_alg3, FPCA_alg5] = runClustering(W_Fair, D_Fair, F, gmale);
[NoPCA_alg1, NoPCA_alg3, NoPCA_alg5] = runClustering(W_norm, D_norm, F, gmale);
[HarshPCA_alg1, HarshPCA_alg3, HarshPCA_alg5] = runClustering(W_harsh, D_harsh, F, gmale);

%% ---------------------------------------------------------------
%  VISUAL PROOF: STRUCTURAL DESTRUCTION
% ---------------------------------------------------------------
fprintf('Generating Structural Visualizations...\n');

% 1. Subsample data for visualization (16,000 is too big to plot connections)
% We take 100 random points from Class 0 and 100 from Class 1
idx_viz_A = find(Group == 0, 100);
idx_viz_B = find(Group == 1, 100);
idx_viz = [idx_viz_A; idx_viz_B];

% Sort by ground truth cluster (Y) to see "Block Structure"
[~, sort_order] = sort(Y(idx_viz)); 
idx_viz_sorted = idx_viz(sort_order);

% Re-compute small graphs for visualization
W_PCA_viz = build_knn_graph(M_PCA(idx_viz_sorted, :), k_nn);
W_Fair_viz = build_knn_graph(M_FairPCA(idx_viz_sorted, :), k_nn);
W_Harsh_viz = build_knn_graph(M_HarshFairPCA(idx_viz_sorted, :), k_nn);

% ---------------------------------------------------------------
%  PLOT 1: THE GEOMETRY (2D Projection)
% ---------------------------------------------------------------
figure('Name', 'Geometry of the Subspace', 'Position', [100, 100, 1200, 400]);

subplot(1,3,1);
gscatter(M_PCA(idx_viz,1), M_PCA(idx_viz,2), Y(idx_viz));
title('Standard PCA Space');
subtitle('distinct clusters likely visible');
xlabel('Dim 1'); ylabel('Dim 2'); legend off;

subplot(1,3,2);
gscatter(M_FairPCA(idx_viz,1), M_FairPCA(idx_viz,2), Y(idx_viz));
title('MW Fair-PCA Space');
subtitle('clusters likely merged/distorted');
xlabel('Dim 1'); ylabel('Dim 2'); legend off;

subplot(1,3,3);
gscatter(M_HarshFairPCA(idx_viz,1), M_HarshFairPCA(idx_viz,2), Y(idx_viz));
title('SDP Fair-PCA Space');
subtitle('strict constraint distortion');
xlabel('Dim 1'); ylabel('Dim 2'); legend off;
%%

% ============================
%        PLOTTING RESULTS
% ============================

krange = 2:1:5;

ulesfontsize = 15;
set(0, 'DefaultAxesFontSize', ulesfontsize);
set(0, 'DefaultTextFontSize', ulesfontsize);
set(0, 'DefaultLineLineWidth', 2.5);

% ---- Plot 1: Alg1 (Standard SC) ----
figure; clf; hold on;

plot(krange, max(NoPCA_alg1, [], 2), 'bp-', 'MarkerSize', 8, ...
    'DisplayName', 'No-PCA Data');
plot(krange, max(PCA_alg1, [], 2), 'go--', 'MarkerSize', 8, ...
    'DisplayName', 'PCA Data');
plot(krange, max(FPCA_alg1, [], 2), 'ms-', 'MarkerSize', 8, ...
    'DisplayName', 'MW-Fair-PCA Data');
plot(krange, max(HarshPCA_alg1, [], 2), 'rx-', 'MarkerSize', 8, ...
    'DisplayName', 'SDP-Fair-PCA Data');
xlabel('k (clusters)');
ylabel('Balance');
title('Impact of PCA on SC');
legend('Location','northwest');
grid on;

% ---- Plot 2: Alg3 (S-Fair-SC) ----
figure; clf; hold on;

plot(krange, max(NoPCA_alg3, [], 2), 'bp-', 'MarkerSize', 8, ...
    'DisplayName', 'No-PCA Data');
plot(krange, max(PCA_alg3, [], 2), 'go--', 'MarkerSize', 8, ...
    'DisplayName', 'PCA Data');
plot(krange, max(FPCA_alg3, [], 2), 'ms-', 'MarkerSize', 8, ...
    'DisplayName', 'MW-Fair-PCA Data');
plot(krange, max(HarshPCA_alg3, [], 2), 'rx-', 'MarkerSize', 8, ...
    'DisplayName', 'SDP-Fair-PCA Data');
xlabel('k (clusters)');
ylabel('Balance');
title('Impact of PCA on S-Fair-SC');
legend('Location','northwest');
grid on;

% ---- Plot 3: Alg5 (AFF-Fair-SMW) ----
figure; clf; hold on;

plot(krange, max(NoPCA_alg5, [], 2), 'bp-', 'MarkerSize', 8, ...
    'DisplayName', 'No-PCA Data');
plot(krange, max(PCA_alg5, [], 2), 'go--', 'MarkerSize', 8, ...
    'DisplayName', 'PCA Data');
plot(krange, max(FPCA_alg5, [], 2), 'ms-', 'MarkerSize', 8, ...
    'DisplayName', 'MW-Fair-PCA Data');

plot(krange, max(HarshPCA_alg5, [], 2), 'rx-', 'MarkerSize', 8, ...
    'DisplayName', 'SDP-Fair-PCA Data');

xlabel('k (clusters)');
ylabel('Balance');
title('Impact of PCA on AFF-Fair-SMW');
legend('Location','northwest');
grid on;


% --- Helper Function for Clustering Loop ---
function [balance1_all, balance3_all, balance5_all] = runClustering(W, D, F, gmale)
    krange = 2:1:5;
    numRuns = 1; % One run is fine for synthetic data

    balance1_all = zeros(length(krange), numRuns);
    balance3_all = zeros(length(krange), numRuns);
    balance5_all = zeros(length(krange), numRuns);

    for run = 1:numRuns
        for i = 1:length(krange)
            k = krange(i);

            [labelsalg1 , ~] = alg1(W, D, k);
            balance1_all(i, run) = computeBalance(labelsalg1, gmale, k);

            [labelsalg3 , ~] = alg3(W, D, F, k);
            balance3_all(i, run) = computeBalance(labelsalg3, gmale, k);

            [labelsalg5, ~] = alg5(W, D, F, k);
            balance5_all(i, run) = computeBalance(labelsalg5, gmale, k);
        end
    end
end