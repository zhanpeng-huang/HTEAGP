clear;clc;
addpath(genpath(cd));

load 'yale.mat';

XX = {full(X{1})',full(X{2})',full(X{3})'};
gt = double(gt); 
viewNum = size(XX,2);
N = size(XX{1},1); 
H = cell(1,viewNum);
I = eye(N);
metric = 'cosine'; 
beta = [1,10,10]';
WW = make_distance_matrix(XX, metric);

lambda_values = [0.1,0.3,0.5,0.7,0.9,1,5,10]; 
K_values = [2,4,6,8,10];

res_ACC = zeros(length(lambda_values), length(K_values));

for i = 1:length(lambda_values)
    for j = 1:length(K_values)
        lambda = lambda_values(i);
        K = K_values(j);
        X = XX;      
        for v = 1:viewNum
            A{v} = make_kNN_dist(WW{v}, K);
            D{v} = diag(sum(A{v},2));
            L{v} = I-((D{v})^(-0.5)*(A{v}+I)*(D{v})^(-0.5));
        
            U_views = cell(viewNum, 1); 
            lambda_views = cell(viewNum, 1); 
            Lambda = cell(viewNum, 1);
            [U_views{v}, Lambda{v}, ~] = svd(L{v}); 
            lambda_views{v} = diag(Lambda{v}); 
        
            X_hat_views = cell(viewNum, 1);
            X_hat_views{v} = U_views{v}' * X{v};
        
            mu_views = cell(viewNum, 1); 
            sigma_views = cell(viewNum, 1); 
            g_theta_views = cell(viewNum, 1); 
        
            mu_views{v} = mean(lambda_views{v}); 
            sigma_views{v} = std(lambda_views{v}); 
            g_theta_views{v} = @(Lambda) exp(-(Lambda(v) - mu_views{v}).^2 ./ (2 * sigma_views{v}^2)); % 高斯滤波器函数
        
            X_smooth_views = cell(viewNum, 1); 
            g_theta_lambda_views{v} = g_theta_views{v}(lambda_views{v});
            Lambda_views{v} = diag(g_theta_lambda_views{v});
            X_smooth_views{v} = U_views{v} * Lambda_views{v} * U_views{v}' * X{v};
            X{v} = X_smooth_views{v};
        
            degrees = sum(A{v}, 2);
            alpha_adaptive = 1 ./ (1 + degrees); 
            P_adaptive = alpha_adaptive .* ((D{v})^(-0.5) * A{v} * (D{v})^(-0.5)) + (1-alpha_adaptive) .* I;
            num_iterations = 5; 
            H_temp = X{v};
            for iter = 1:num_iterations
                H_temp = P_adaptive * H_temp;
            end
            H{v} = H_temp;
            H{v} = H{v}';
        end

        [My_result, S, C] = HTEAGP(H, gt, lambda, beta);
        res_ACC(i, j) = My_result(1);
        fprintf('Processing lambda=%.2f, K=%d, ACC=%.4f\n', lambda, K, My_result(1));
    end
end




