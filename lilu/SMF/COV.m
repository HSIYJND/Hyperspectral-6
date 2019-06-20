function [C] = COV(M)
% HYPERCOV Computes the covariance matrix
%COV����������Э�������
% hyperCorr compute the sample covariance matrix of a 2D matrix.
%hypercorr���������˶�ά���������Э�����
%
% Usage
%   [C] = hyperCorr(M)
%
% Inputs
%   M - 2D matrix 
% Outputs
%   C - Sample covariance matrix

[p, N] = size(M);%pΪ������NΪ����
% Remove mean from data%��������ɾ����ֵ
u = mean(M.').';%M.'��M��ת�þ���
                %��M��ÿһ�е�ƽ��ֵ��uΪһ��p��1�е��о���
M = M - repmat(u, 1, N);%repmatΪ�ظ����鸱��������repmat��u,1,N��Ϊp��N�о���
                        %�൱��M������ÿһ�е�ÿһ��������ȥ���е�ƽ��ֵ
C = (M*M.')/N;
