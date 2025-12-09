clear; clc; close all;

[M, A, B,Y] = creditProcess();

n_A1 = length(A);
n_B1 = length(B);
Group = [zeros(n_A1,1); ones(n_B1,1)]; % 0 = Male, 1 = Female

d = 21;

featureNum = 2; % This is gonna be the amount of protected groups were gonna capture


coeff = pca(M);
loss_A = zeros(featureNum,1);
loss_B = zeros(featureNum,1);
lossFair_A = zeros(featureNum,1);
lossFair_B = zeros(featureNum,1);
lossFair_max = zeros(featureNum,1);

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
    [P_fair, z(ell), P_last, z_last(ell)] = mw(A, B, ell, eta, T);

    % Smart selection
    if z(ell) < z_last(ell)
        P_smart = P_fair;
    else
        P_smart = P_last;
    end

    % Round to projection
    P_smart = eye(size(P_smart,1)) - sqrtm(eye(size(P_smart,1)) - P_smart);

    % Fair reconstruction
    approxFair_A = A * P_smart;
    approxFair_B = B * P_smart;

    % Fair reconstruction loss
    lossFair_A(ell) = loss(A, approxFair_A, ell)/size(A,1);
    lossFair_B(ell) = loss(B, approxFair_B, ell)/size(B,1);
    lossFair_max(ell) = max([lossFair_A(ell), lossFair_B(ell)]);

end

% ---------------------------------------------------------------
%  PLOT PCA VS FAIR PCA LOSS
% ---------------------------------------------------------------

figure; hold on;
plot(1:featureNum, loss_A,'gx-','LineWidth',2);
plot(1:featureNum, loss_B,'ro-','LineWidth',2);
plot(1:featureNum, lossFair_max,'b--o','LineWidth',2);

title('PCA vs Fair-PCA Reconstruction Loss (Scenario 3)');
legend('Male (PCA)','Female (PCA)','Fair PCA (max loss)');
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
b = bar(1:d, [importance_pca, importance_fair,importance_harsh_fair]);
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
%  CLASSIFIERS
% ---------------------------------------------------------------
% We use 'Uniform' prior to tell the classifier to treat classes equally
% even though the data is imbalanced.
mdl_pca = fitcdiscr(Xtrain_pca, ytrain);
mdl_fpca = fitcdiscr(Xtrain_fpca, ytrain);
mdl_harshfpca = fitcdiscr(Xtrain_harshfpca, ytrain);


% PCA model prediction
[~, score_pca]  = predict(mdl_pca, Xtest_pca);
prob_pca = score_pca(:,2);   
yhat_pca  = prob_pca  >= 0.5;

% Fair PCA model prediction
[~, score_fpca] = predict(mdl_fpca, Xtest_fpca);
prob_fpca = score_fpca(:,2); 
yhat_fpca = prob_fpca >= 0.5;

% Fair SDPPCA model prediction
[~, score_harshfpca] = predict(mdl_harshfpca, Xtest_harshfpca);
prob_harshfpca = score_harshfpca(:,2); 
yhat_harshfpca = prob_harshfpca >= 0.5;

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

