clc;
clear all; 
[M, A, B] = creditProcess();

featureNum = 5;

% --- Setup for Vanilla PCA ---
coeff = pca(M);
recons_A = zeros(featureNum,1);
recons_B = zeros(featureNum,1);
reconsAhat = zeros(featureNum, 1);
reconsBhat = zeros(featureNum, 1);

% --- Setup for Fair PCA (MW) ---
reconsFair_A_MW = zeros(featureNum, 1);
reconsFair_B_MW = zeros(featureNum, 1);
z = zeros(featureNum, 1);
z_smart = zeros(featureNum, 1);
eta = 20;
T = 5;

% --- Setup for Fair PCA (SDP) ---
reconsFair_A_SDP = zeros(featureNum, 1);
reconsFair_B_SDP = zeros(featureNum, 1);

% --- NEW: Arrays to store the runtimes ---
time_sdp = zeros(featureNum, 1);
time_mw = zeros(featureNum, 1);

fprintf('Starting comparison for d=1 to %d...\n', featureNum);

for ell=1:featureNum
    fprintf('--- Processing for d = %d ---\n', ell);
    
    % --- 1. Vanilla PCA (Your original code) ---
    P = coeff(:,1:ell)*transpose(coeff(:,1:ell));
    approx_A = A*P;
    approx_B = B*P;
    recons_A(ell) = re(A, approx_A)/size(A, 1);
    recons_B(ell) = re(B, approx_B)/size(B, 1);
    
    Ahat = optApprox(A, ell);
    reconsAhat(ell) = re(A, Ahat)/size(A, 1);
    Bhat = optApprox(B, ell);
    reconsBhat(ell) = re(B, Bhat)/size(B, 1);
    
    % --- 2. Fair PCA (SDP Method) ---
    fprintf('Running SDP (cvx)... ');
    tic; % Start SDP timer
    P_sdp = fair_pca(A, B, ell); % Run the SDP function
    time_sdp(ell) = toc; % Stop timer and store time
    fprintf('Done in %.2f seconds.\n', time_sdp(ell));
    
    % Calculate reconstruction error for SDP
    approxFair_SDP_A = A * P_sdp;
    approxFair_SDP_B = B * P_sdp;
    reconsFair_A_SDP(ell) = re(approxFair_SDP_A, A)/size(A, 1);
    reconsFair_B_SDP(ell) = re(approxFair_SDP_B, B)/size(B, 1);

    % --- 3. Fair PCA (MW Method) ---
    fprintf('Running MW... ');
    tic; % Start MW timer
    [P_fair, z(ell), P_last, z_last(ell)] = mw(A, B, ell, eta/ell, T);
    time_mw(ell) = toc; % Stop timer and store time
    fprintf('Done in %.2f seconds.\n', time_mw(ell));

    % (Your original logic for processing MW results)
    if z(ell) < z_last(ell)
        P_smart = P_fair;
    else
        P_smart = P_last;
    end
    P_smart = eye(size(P_smart,1)) - sqrtm(eye(size(P_smart,1))-P_smart);
    
    approxFair_A = A * P_smart;
    approxFair_B = B * P_smart;
    reconsFair_A_MW(ell) = re(approxFair_A, A)/size(A, 1);
    reconsFair_B_MW(ell) = re(approxFair_B, B)/size(B, 1);
end

fprintf('--- All runs complete ---\n');

% --- Plot 1: Reconstruction Errors ---
checkpoints = 1:featureNum;
figure;
plot(checkpoints, recons_A,'rx-', checkpoints, recons_B, 'bx-', ...
     checkpoints, reconsFair_A_MW,'r*-', checkpoints, reconsFair_B_MW,'b*-', ...
     checkpoints, reconsFair_A_SDP,'g*-', checkpoints, reconsFair_B_SDP,'c*-');
title('Average reconstruction error (ARE)')
legend('ARE A PCA','ARE B PCA', 'ARE A (MW)', 'ARE B (MW)', 'ARE A (SDP)', 'ARE B (SDP)')
xlabel('Number of features (d)')
ylabel('Average reconstruction error (ARE)')

% --- Plot 2: Runtimes ---
figure;
plot(checkpoints, time_sdp, 'm-s', 'LineWidth', 2);
hold on;
plot(checkpoints, time_mw, 'g-o', 'LineWidth', 2);
hold off;
title('Algorithm Runtimes vs. Target Dimension');
xlabel('Number of features (d)');
ylabel('Time (seconds)');
legend('SDP (cvx)', 'Multiplicative Weights (mw)');
set(gca, 'YScale', 'log'); % Use log scale, times will be very different!