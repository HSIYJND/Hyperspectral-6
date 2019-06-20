tic
clear;
clc;


load('D:\高光谱\识别\新建文件夹\数据\test2.mat');

sizeRR=size(RR);
sizeY=size(Y);
t=mean(T,2);%返回包含每一行均值的列向量
result=SMF(Y,t);
%下面的for循环是将result归一化到（0，1） 
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

r=reshape(result,sizeRR(2),sizeRR(1));%将result数据重构为一个100*100矩阵
                                      %result是广义似然比表达式的值
r=r';%对r矩阵求转置
threshold=1;%设定阈值
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
[c,d]=JS(S,Q);%矩阵Q中元素有42个1，其余全部为0
              %c表示S中1的个数，即大于阈值的目标个数。
              %c表示虚警，将背景像元误认成目标像元
              %d表示漏检，将目标像元误认成背景像元
A(k,:)=[c d];
threshold=threshold-0.0001;
end
%此时A矩阵为10001*2的矩阵
%第一列为在阈值为(1-0.0001*(k-1))下虚警数目。
%第二列为在阈值为(1-0.0001*(k-1))下漏检数目。
for j=1:10001,
    A(j,2)=42-A(j,2);%42是原图像目标总像素数目。
                     %此时A的第二列表示正确检出目标像元数
end
B=zeros(10001,2);
B(:,1)=9958;%B矩阵是10001行2列矩阵，第一列全部为9958，9958是原图像背景总像素
B(:,2)=42;%B矩阵是10001行2列矩阵，第二列全部为42，42是原图像目标总像素数目
C_S=A./B;%A矩阵的第1（2）列除以B矩阵第1（2）列成为C矩阵的第1（2）列，
x=C_S(:,1);%横坐标为虚警率
%虚警率为在一次判决中被判定为目标的背景像元的个数与背景像元总数的比值
y=C_S(:,2);%纵坐标为已检测率
%检测率指一次判决中被正确识别的目标像元个数与目标像元总数的比值
plot(x,y)
AUC=AUC(C_S)
toc

