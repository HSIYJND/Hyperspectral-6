function [results] =SMF(M, t)
% TODO Fix this
% HYPERACE Performs the adaptive cosin/coherent estimator algorithm
%   Performs the adaptive cosin/coherent estimator algorithm for target
% detection.
%HYPERACE执行的是自适应余弦/相干估计算法来进行目标检测
% Usage
%   [results] = hyperAce(M, S)
% Inputs
%   M - 2d matrix of HSI data (p x N)
%   S - 2d matrix of target endmembers (p x q)
% Outputs
%   results - vector of detector output (N x 1)
%
% References
%   X Jin, S Paswater, H Cline.  "A Comparative Study of Target Detection
% Algorithms for Hyperspectral Imagery."  SPIE Algorithms and Technologies
% for Multispectral, Hyperspectral, and Ultraspectral Imagery XV.  Vol
% 7334.  2009.


[p, N] = size(M);
% Remove mean from data
u = mean(M.').';%求M的每一行的平均值，u为一个p行1列的列矩阵
M = M - repmat(u, 1, N); %相当于M矩阵中每一行的每一个数都减去该行的平均值
t = t - u;

R_hat = COV(M);
G = pinv(R_hat);%求R_hat的逆矩阵

results = zeros(1, N);
tmp = t.'*G*t;%tmp为一个常数
for k=1:N
    x = M(:,k);
    results(k) = (x.'*G*t) / tmp;
end
