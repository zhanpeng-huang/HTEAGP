% HTEAGP - High-order tensor based multi-view clustering via enhanced adaptive graph propagation
% The method iteratively updates several variables (such as Z, E, Y, G, W, etc.) to optimize the clustering result, 
% and finally applies spectral clustering to obtain the cluster labels.
%
% Inputs:
%   X        - A cell array containing data from multiple views, where each cell is a matrix representing data from a specific view.
%   gt       - Ground truth labels used for clustering evaluation.
%   lambda   - Regularization parameter controlling sparsity.
%   beta     - Parameter used for weighting.
%
% Outputs:
%   My_result  - Clustering performance metrics (e.g., ACC, NMI, F1, ARI).
%   S          - The similarity matrix updated during the clustering process.
%   C          - The cluster labels from the final clustering result.
%   kerNS      - The similarity matrix after spectral clustering.
%
% Algorithm Steps:
% 1. Data preprocessing: Normalize the data for each view.
% 2. Initialize variables: Initialize Z, W, G, E, Y for each view.
% 3. Iterative optimization:
%    a. Update Z (the latent representation for each view).
%    b. Update E (reconstruction error).
%    c. Update Y (Lagrange multipliers).
% 4. Update G (weighted information)、W (auxiliary variables)、mu and rho.
% 5. Perform spectral clustering: Use the similarity matrix S for spectral clustering and calculate clustering performance metrics.
% 6. Check convergence: If convergence is met, stop the iterations and output the final clustering result.
function [My_result,S,C,kerNS]=HTEAGP(X,gt,lambda,beta)

cls_num = size(unique(gt),1);
num_views = size(X,2);
N = size(X{1},2); 
% Data preprocessing
for v = 1:num_views
    X{v}=NormalizeData(X{v});
end

% Initialize and Settings
for k=1:num_views
    Z{k} = zeros(N,N);
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N);
end
K = num_views;
sX = [N, N, K];
epson = 1e-10; 
mu = 10e-5; 
max_mu = 10e12;  
pho_mu = 2; 
rho = 0.0001; 
max_rho = 10e10;  
pho_rho = 2;
converge_Z=[]; 
converge_Z_G=[];
iter = 0;
Isconverg = 0; 
num = 0;
while(Isconverg == 0)
    fprintf('---------processing iter %d--------\n', iter+1);
    num = num + 1;
    for k=1:K
        % Update Z^k  
        tmp = (X{k}'*Y{k} + mu*X{k}'*X{k} - mu*X{k}'*E{k} - W{k})./rho +  G{k};
        Z{k}=pinv(eye(N,N)+ (mu/rho)*X{k}'*X{k})*tmp;
        
        % Update E^k   
        F = [X{1}-X{1}*Z{1}+Y{1}/mu;X{2}-X{2}*Z{2}+Y{2}/mu;X{3}-X{3}*Z{3}+Y{3}/mu];
        
        [Econcat] = solve_l1l2(F,lambda/mu);
        E{1} = Econcat(1:size(X{1},1),:);
        E{2} = Econcat(size(X{1},1)+1:size(X{1},1)+size(X{2},1),:);
        E{3} = Econcat(size(X{1},1)+size(X{2},1)+1:size(X{1},1)+size(X{2},1)+size(X{3},1),:);
       
        % Update Y^k  
        Y{k} = Y{k} + mu*(X{k}-X{k}*Z{k}-E{k});
    end
    
    %% Update G 
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:); 
    w = W_tensor(:); 

    [g, objV] = wshrinkObj_weight(z + 1/rho*w,beta/rho,sX,0,3);

    G_tensor = reshape(g, sX); 
    % Update W 
    w = w + rho*(z - g);
    
    % Record the iteration information
    history.objval(iter+1) = objV;
    % Coverge condition
    max_Z=0;
    max_Z_G=0;
    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z= norm(X{k}-X{k}*Z{k}-E{k},inf);
            fprintf('norm_Z %7.10f', history.norm_Z);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        G{k} = G_tensor(:,:,k); 
        W_tensor = reshape(w, sX); 
        W{k} = W_tensor(:,:,k); 
        if (norm(Z{k}-G{k},inf)>epson)
            history.norm_Z_G= norm(Z{k}-G{k},inf); 
            Isconverg = 0;
            max_Z_G=max(max_Z_G, history.norm_Z_G); 
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];
    if (iter>200)
        Isconverg = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);

 
    S = 0;
    for k=1:K
        S = S + abs(Z{k}');  
    end
    [C,kerNS] = SpectralClustering(S./K, cls_num);
    [result(:,:)] = ClusteringMeasure1(gt, C);
  
end

My_result = result(:,:); % ACC NMI F1 ARI


