function A=AUC(C)

x=C(:,1);%x为矩阵的第一列
y=C(:,2);%y为矩阵的第二列
S=0;
for i=2:size(x)
    if y(i)>y(i-1)
        S=S+(((x(i)-x(i-1))*(y(i)-y(i-1))/2)+(y(i-1)*(x(i)-x(i-1))));
    else
        S=S+((x(i)-x(i-1))*y(i));
    end
end
A=S;
