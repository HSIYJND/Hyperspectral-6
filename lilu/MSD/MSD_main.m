tic
clear;
clc;

load('D:\高光谱\识别\新建文件夹\数据\test1.mat');

sizeRR=size(RR);
sizeY=size(Y);
t=mean(T,2);
[V_b,V_t]=extract(Y_S,target_samples);
result=MSD(Y,V_b,V_t);

MAX=max(result);
MIN=min(result);
l=1/(MAX-MIN);
for j=1:sizeY(2)
    if result(j)==MIN
        result(j)=0;
    elseif result(j)==MAX
        result(j)=1;
    else
        result(j)=(result(j)-MIN)*l;
    end
end

r=reshape(result,sizeRR(2),sizeRR(1));
r=r';
threshold=1;
for k=1:10001
    for i=1:sizeRR(1)
        for j=1:sizeRR(2)
            if r(i,j)>threshold
                S(i,j)=1;
            else
                S(i,j)=0;
             end 
        end
    end
[c,d]=JS(S,Q);
A(k,:)=[c d];
threshold=threshold-0.0001;
end

for j=1:10001,
    A(j,2)=97-A(j,2);
end
B=zeros(10001,2);
B(:,1)=9903;
B(:,2)=97;
C_M=A./B;
x=C_M(:,1);
y=C_M(:,2);
plot(x,y)
AUC=AUC(C_M)
toc
