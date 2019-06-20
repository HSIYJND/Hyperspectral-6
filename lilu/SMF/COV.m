function [C] = COV(M)
% HYPERCOV Computes the covariance matrix
%COV函数计算了协方差矩阵
% hyperCorr compute the sample covariance matrix of a 2D matrix.
%hypercorr函数计算了二维矩阵的样本协方差函数
%
% Usage
%   [C] = hyperCorr(M)
%
% Inputs
%   M - 2D matrix 
% Outputs
%   C - Sample covariance matrix

[p, N] = size(M);%p为行数，N为列数
% Remove mean from data%从数据中删除均值
u = mean(M.').';%M.'是M的转置矩阵
                %求M的每一行的平均值，u为一个p行1列的列矩阵
M = M - repmat(u, 1, N);%repmat为重复数组副本函数，repmat（u,1,N）为p行N列矩阵
                        %相当于M矩阵中每一行的每一个数都减去该行的平均值
C = (M*M.')/N;
