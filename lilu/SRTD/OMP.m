function [v,r]=OMP(D,X,L)
% �������: 
% D - ���걸�ֵ䣬ע�⣺�����ֵ�ĸ��б��뾭���˹淶�� 
% X - �ź�
%L - ϵ���з���Ԫ���������ֵ����ѡ��Ĭ��ΪD���������ٶȿ������� 
% �������: 
% v - ϡ��ϵ�� 
%r- �в�
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
