function [v,r]=OMP(D,X,L)
% 输入参数: 
% D - 过完备字典，注意：必须字典的各列必须经过了规范化 
% X - 信号
%L - 系数中非零元个数的最大值（可选，默认为D的列数，速度可能慢） 
% 输出参数: 
% v - 稀疏系数 
%r- 残差
[rows,~]=size(X);
d=D./repmat(sqrt(sum(D.^2,1)),rows,1);
P=size(X,2);
K=size(D,2); 
count=1;
for k=1:1:P, 
    a=[]; x=X(:,k); residual=x; indx=zeros(L,1); 
    for j=1:1:L,
        proj=d'*residual; 
        [~,pos]=max(abs(proj)); 
        pos=pos(1); 
        indx(j)=pos; 
        a=pinv(d(:,indx(1:j)))*x;
        residual=x-d(:,indx(1:j))*a;
        if (sum(residual.^2) < 1e-6||count>50) 
            break; 
        end
        count=count+1;
    end; 
    temp=zeros(K,1);
    temp(indx(1:j))=a;
    v(:,k)=sparse(temp);
    r=residual;
   
end;
%r=norm(r,2);
