% wshrinkObj_weight - Perform weighted shrinkage operation using soft thresholding
%
% This function implements a weighted shrinkage operation, which is commonly used in signal processing for sparse signal recovery 
% or dimensionality reduction. The shrinkage is performed on the singular values of the Fourier transformed tensor slices, 
% and soft thresholding is applied to enforce sparsity.
%
% Inputs:
%   x       - A vector representing the input data that will be reshaped into a 3D tensor.
%   rho     - A scalar or vector that controls the shrinkage (penalty) level applied to the singular values.
%   sX      - A vector specifying the dimensions of the 3D tensor (e.g., [n1, n2, n3]).
%   isWeight - A binary flag (0 or 1) indicating whether a weight is applied during the shrinkage process.
%   mode    - A scalar (1, 2, or 3) specifying the slicing mode for the tensor (default: 1). 
%             - 1: lateral slice, 2: front slice, 3: top slice.
%
% Outputs:
%   x       - The reshaped and processed tensor as a vector after applying the shrinkage operation.
%   objV    - The objective value (sum of the singular values after shrinkage), which can be used for convergence analysis.
%
% Algorithm Steps:
% 1. The input vector `x` is reshaped into a 3D tensor using the provided dimensions `sX`.
% 2. The tensor slices are transformed using the Fourier Transform (FFT).
% 3. Singular Value Decomposition (SVD) is applied to the Fourier-transformed slices.
% 4. Soft thresholding is applied to the singular values to enforce sparsity, either with or without weights (based on `isWeight`).
% 5. The tensor is transformed back to the original space using the inverse Fourier Transform (IFFT).
% 6. The reshaped and processed tensor is flattened back into a vector.

function [x,objV] = wshrinkObj_weight(x,rho,sX, isWeight,mode)
%rho is the weighted vector
if isWeight == 1
    %     C = 2*sqrt(2)*sqrt(sX(3)*sX(2));
    C = sqrt(sX(3)*sX(2));
end
if ~exist('mode','var')
    % mode = 1 --> lateral slice
    % mode = 2 --> front slice
    % mode = 3 --> top slice
    mode = 1;
end

X=reshape(x,sX);
if mode == 1
    Y=X2Yi(X,3);
elseif mode == 3
    Y=shiftdim(X, 1); 
else
    Y = X;
end


Yhat = fft(Y,[],3);

objV = 0;
if mode == 1
    n3 = sX(2);
elseif mode == 3
    n3 = sX(1);
else
    n3 = sX(3);
end

if isinteger(n3/2)
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        
        if isWeight
            weight = C./(diag(shat) + eps);
            tau = rho*weight;
            shat = soft(shat,diag(tau));
        else
            tau = diag(rho);
            shat = max(shat - tau,0);
        end
        
        objV = objV + sum(shat(:));
        Yhat(:,:,i) = uhat*shat*vhat';
        if i > 1
            Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
    [uhat,shat,vhat] = svd(full(Yhat(:,:,endValue+1)),'econ'); 
    if isWeight
        weight = C./(diag(shat) + eps);
        tau = rho*weight;
        shat = soft(shat,diag(tau));
    else
        tau = diag(rho);
        shat = max(shat - tau,0);

    end
    
    objV = objV + sum(shat(:));
    Yhat(:,:,endValue+1) = uhat*shat*vhat';
else
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        
        if isWeight
            weight = C./(diag(shat) + eps);
            tau = rho*weight;
            shat = soft(shat,diag(tau));
        else
            tau = diag(rho); 
            shat = max(shat - tau,0);
            

        end
        objV = objV + sum(shat(:));
        Yhat(:,:,i) = uhat*shat*vhat'; 
        if i > 1 
            Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
end


Y = ifft(Yhat,[],3);

if mode == 1
    X = Yi2X(Y,3);
elseif mode == 3
    X = shiftdim(Y, 2);
else
    X = Y;
end

x = X(:);

end
