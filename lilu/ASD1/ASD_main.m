tic
clear;
clc;


load('C:\Users\Lilu\Documents\MATLAB\test1.mat');


[row,column]=size(Y);
S=zeros(100,100);
l=[];



threshold=1;



[results_ASD] = ASD(Y,T_5);

result=abs(results_ASD);
f=max(result);
g=min(result);
s=1/(f-g);
for j=1:column
    if result(j)==g
        r(j)=0;
    elseif result(j)==f
        r(j)=1;
    else
        r(j)=(result(j)-g)*s;
    end
end

for i=1:10001
    for k=1:column,
        if r(1,k)>threshold
            l(1,k)=1;
        else
            l(1,k)=0;
        end
    end



S=reshape(l,100,100);
S=S';
[c,d]=JS(S,Q);
A(i,:)=[c d];

threshold=threshold-0.0001;
end
for j=1:10001,
    A(j,2)=21-A(j,2);
end
B=zeros(10001,2);
B(:,1)=9979;
B(:,2)=21;
C_M=A./B;
x=C_M(:,1);
y=C_M(:,2);
plot(x,y);
AUC=AUC(C_M)
toc

