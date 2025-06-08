clear;clc;
addpath(genpath(cd));

load '3-sources.mat';
gt = truth;
X = {bbc,guardian,reuters};

viewNum = size(X,2);
N = size(X{1},1); 
H = cell(1,viewNum);
I = eye(N);
metric = 'cosine'; 
WW = make_distance_matrix(X, metric);
beta = [10,10,10]';

U_views = cell(viewNum, 1); % feature vector
lambda_views = cell(viewNum, 1); % eigenvalue
mu_views = cell(viewNum, 1);  % center frequency
sigma_views = cell(viewNum, 1);  % the bandwidth of the filter
g_theta_views = cell(viewNum, 1); % Gaussian filter
X_smooth_views = cell(viewNum, 1); % Smoothed signal
g_theta_lambda_views = cell(viewNum, 1);
Lambda_views = cell(viewNum, 1);
degrees = cell(viewNum, 1); 
alpha_adaptive = cell(viewNum, 1); 
P_adaptive = cell(viewNum, 1); 
A = cell(viewNum, 1);
D = cell(viewNum, 1);
L = cell(viewNum, 1);
X_hat_views = cell(viewNum, 1);

lambda_values = [0.1,0.3,0.5,0.7,0.9,1,5,10]; 
K_values = [2,4,6,8,10];

res_ACC = zeros(length(lambda_values), length(K_values));

for i = 1:length(lambda_values)
    for j = 1:length(K_values)
      
        lambda = lambda_values(i);
        K = K_values(j);

        X = {bbc,guardian,reuters};

        for v = 1:viewNum
            A{v} = make_kNN_dist(WW{v}, K);
            D{v} = diag(sum(A{v},2));
            L{v} = I-((D{v})^(-0.5)*(A{v}+I)*(D{v})^(-0.5));
            % feature decomposition
            [U_views{v}, ~, ~] = svd(L{v}); 
            lambda_views{v} = diag(L{v}); 
            % fourier Transform
            X_hat_views{v} = U_views{v}' * X{v};
            
            % Gaussian filter
            mu_views{v} = mean(lambda_views{v}); 
            sigma_views{v} = std(lambda_views{v}); 
            g_theta_views{v} = @(lambda) exp(-(lambda - mu_views{v}).^2 ./ (2 * sigma_views{v}^2));
            
            g_theta_lambda_views{v} = g_theta_views{v}(lambda_views{v});
            Lambda_views{v} = diag(g_theta_lambda_views{v});
            X_smooth_views{v} = U_views{v} * Lambda_views{v} * X_hat_views{v};
            X{v} = X_smooth_views{v};
            
            % adaptive graph propagation
            degrees{v} = sum(A{v}, 2);
            alpha_adaptive{v} = 1 ./ (1 + degrees{v});
            P_adaptive{v} = alpha_adaptive{v} .* ((D{v})^(-0.5) * A{v} * (D{v})^(-0.5)) + (1-alpha_adaptive{v}) .* I;
        end
        
        P = zeros(N);
        for v = 1:viewNum
            P = P + P_adaptive{v};
        end
        P = P/viewNum;
        
        for v = 1:viewNum
            num_iterations = 5;
            for iter = 1:num_iterations
                X{v} = P * X{v};
            end
            H{v} = X{v}';
        end
        
        [My_result, S, C] = HTEAGP(H, gt, lambda, beta);
        res_ACC(i, j) = My_result(1);
        fprintf('Processing lambda=%.2f, K=%d, ACC=%.4f\n', lambda, K, My_result(1));
    end
end



