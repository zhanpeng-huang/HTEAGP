% Contingency - Compute the contingency table (also known as confusion matrix or co-occurrence matrix)
%
% This function calculates the contingency table (confusion matrix) between two clustering results (or classification results), 
% which is used to compare how the two different clustering solutions (or classification assignments) agree with each other.
% A contingency table is a useful tool for evaluating the performance of clustering algorithms.
%
% Inputs:
%   Mem1   - A column vector representing the first clustering result. 
%            Each element corresponds to the cluster label assigned to an entity.
%   Mem2   - A column vector representing the second clustering result. 
%            Each element corresponds to the cluster label assigned to an entity.
%
% Outputs:
%   Cont   - The contingency matrix (confusion matrix) where each entry Cont(i,j) represents the number of entities
%            assigned to cluster i in Mem1 and cluster j in Mem2.
%
% The contingency matrix is often used to evaluate clustering results and to compute various clustering performance metrics 
% such as the Rand Index, Adjusted Rand Index, etc.
%
% Example:
%   Mem1 = [1, 2, 1, 1, 2];
%   Mem2 = [1, 2, 2, 1, 1];
%   Cont = Contingency(Mem1, Mem2);
%   % Cont will contain the contingency table comparing Mem1 and Mem2.
%
% Author: David Corney (2000)
% Email: D.Corney@cs.ucl.ac.uk
%


function Cont=Contingency(Mem1,Mem2)

if nargin < 2 | min(size(Mem1)) > 1 | min(size(Mem2)) > 1
   error('Contingency: Requires two vector arguments')
   return
end

Cont=zeros(max(Mem1),max(Mem2));

for i = 1:length(Mem1);
   Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
end
