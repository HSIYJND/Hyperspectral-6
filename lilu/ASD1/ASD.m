function [results] = ASD(M,t)
% HYPERAMSD Adaptive matched subspace detector (AMSD) algorithm
%   Performs the adaptive matched subspace detector (AMSD) algorithm for
% target detection
%
% Usage
%   [results] = hyperAmsd(M, U, target)
% Inputs
%   M - 2d matrix of HSI data (p x N)
%   B - 2d matrix of background endmebers (p x q)
%   target - target of interest (p x 1)
% Outputs
%   results - vector of detector output (N x 1)
%
% References
%   Joshua Broadwater, Reuven Meth, Rama Chellappa.  "A Hybrid Algorithms
% for Subpixel Detection in Hyperspectral Imagery."  IGARSS 004. Vol 3.
% September 2004.

[p, n] = size(M);
% C=COV(M);
% Z=pinv(C);
% MEAN=mean(M,2);
% u=M-repmat(MEAN,1,n);

% I = eye(p);
% P_B=I-(b*pinv(b));
% X=P_B*M;
% C=COV(X);
u=mean(M,2);
P_T=t-repmat(u,1,size(t,2));
Z=inv(COV(M));

% Z=inv(b);

results = zeros(1, n);
tmp = Z*P_T*pinv(P_T'*Z*P_T)*P_T'*Z;
for k=1:n
    x = M(:,k);
    % Equation 16
    results(k) = (x'*tmp*x) / (x'*Z*x);
end