% make_kNN_dist - Construct a k-nearest neighbor (k-NN) graph from a distance matrix
%
% This function takes a distance matrix and builds a k-nearest neighbor (k-NN) graph using Gaussian kernel. 
% It identifies the k closest points for each data point and computes the corresponding similarity values using a Gaussian kernel. 
% The function returns a sparse adjacency matrix representing the k-NN graph and the indices of the neighbors for each data point.
%
% Inputs:
%   D       - A distance matrix of size n x n, where n is the number of data points.
%   knn     - The number of nearest neighbors to consider for each data point.
%
% Outputs:
%   A       - A sparse adjacency matrix of size n x n, representing the similarity (Gaussian kernel) between data points based on their k-NN.
%   idx     - A column vector containing the indices of the k-nearest neighbors for each data point.
%
% Algorithm Steps:
% 1. Find the k-nearest neighbors for each data point using the `mink_new` function. 
%    This returns the distances (A) and the indices (idx) of the k-nearest neighbors.
% 2. Convert the distance matrix A and the index matrix idx into column vectors for easier processing.
% 3. Remove diagonal elements (self-similarity) from the k-NN distance matrix and its corresponding index matrix, 
%    as the distance to the same point is not required.
% 4. Compute the average distance (sigma) and use it to apply a Gaussian kernel function to the distance values.
% 5. Construct a sparse adjacency matrix where the entries represent the similarity between the data points.
% 6. Make the adjacency matrix symmetric, ensuring that the graph is undirected (Aij = Aji).

function [A, idx] = make_kNN_dist(D, knn) 
n = size(D, 1);

[A, idx] = mink_new(D, knn, 2, 'sorting', false); 
A = A(:);

rowidx = repmat((1:n)', knn, 1);
idx = idx(:);
non_diag = rowidx ~= idx;
rowidx = rowidx(non_diag);
idx = idx(non_diag);
A = A(non_diag); 
sigma = mean(sqrt(A)); 
A = exp(-A/(2*sigma^2)); 

A = sparse(rowidx, idx, A, n, n); 
A = (A + A')/2; 

end