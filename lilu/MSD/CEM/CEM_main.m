tic
clear;
clc;


% load('C:\Users\Lilu\Documents\MATLAB\test1.mat');

sizeRR=size(RR);
sizeY=size(Y);
t=mean(T,2);
result=CEM(Y,t);

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
for k=1:10001,
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

for i=1:10001,
    A(i,2)=42-A(i,2);
end
B=zeros(10001,2);
B(:,1)=9958;
B(:,2)=42;
C_C=A./B;
x=C_C(:,1);
y=C_C(:,2);
plot(x,y)
AUC=AUC(C_C)
toc


