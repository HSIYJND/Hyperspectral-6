tic
clear;
clc;

load('G:\资料\新建文件夹\数据\test1.mat');


winlenth=15;
SizeI=3;
RRExpand=expand(RR,winlenth);
[l1,l2,l3]=size(RRExpand);
sizeRR=size(RR);
S=zeros(sizeRR(1),sizeRR(2));
D_t=normalize_data(T);

L=1;





for i=(winlenth+1)/2:(l1-(winlenth-1)/2),
    for j=(winlenth+1)/2:(l2-(winlenth-1)/2),
        [B,D_b,x]=extract(RRExpand,i,j,winlenth,SizeI);
        [v,alpha]=OMP(D_b,x,L);
        r_b(i,j)=norm(alpha,2);
        [v,beta]=OMP(D_t,x,L);
        r_t(i,j)=norm(beta,2);
    end
end

r_b=r_b((winlenth+1)/2:(l1-(winlenth-1)/2),(winlenth+1)/2:...
(l2-(winlenth-1)/2));
r_t=r_t((winlenth+1)/2:(l1-(winlenth-1)/2),(winlenth+1)/2:...
(l2-(winlenth-1)/2));

for i=1:sizeRR(1),
    for j=1:sizeRR(2),
        result(i,j)=r_b(i,j)-r_t(i,j);
    end
end
temp=reshape(result,1,sizeRR(1)*sizeRR(2));
MAX=max(temp);
MIN=min(temp);
l=1/(MAX-MIN);
for i=1:sizeRR(1)*sizeRR(2)
    if temp(i)==MIN
        temp(i)=0;
    elseif temp(i)==MAX
        temp(i)=1;
    else
        temp(i)=(temp(i)-MIN)*l;
    end
end
r=(reshape(temp,sizeRR(2),sizeRR(1)));
threshold=1;
for k=1:10001
    for i=1:sizeRR(1)
        for j=1:sizeRR(2)
            if r(i,j)>threshold
                S(i,j)=1;
            end
        end
    end
[c,d]=JS(S,Q);
A(k,:)=[c d];
threshold=threshold-0.0001;
end

for l=1:10001,
    A(l,2)=58-A(l,2);
end

B=zeros(10001,2);
B(:,1)=9942;
B(:,2)=58;
C_STD=A./B;
x=C_STD(:,1);
y=C_STD(:,2);
plot(x,y)
AUC=AUC(C_STD)
toc
