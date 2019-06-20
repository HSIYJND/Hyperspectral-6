tic
clear;
clc;

load('C:\Users\Lilu\Documents\MATLAB\test1.mat');

sizeRR=size(RR);
sizeY=size(Y);
t=mean(T,2);
result=SFC(Y,t);

threshold=1; 
 for k=1:10001,
    for i=1:sizeY(2)
       
            if result(i)>threshold
                S(i)=1;
            else
                S(i)=0;
            end
        
    end
    S=reshape(S,sizeRR(2),sizeRR(1));
    S=S';
 [c,d]=JS(S,Q);
 A(k,:)=[c d];
 threshold=threshold-0.0001;
 end
 
for j=1:10001
    A(j,2)=58-A(j,2);
end

B=zeros(10001,2);
B(:,1)=9942;
B(:,2)=58;
C_SFC=A./B;
x=C_SFC(:,1);
y=C_SFC(:,2);
plot(x,y)
AUC=AUC(C_SFC)
toc
