function y= SFC(Y,T)
t=mean(T,2);
[m,n]=size(Y);
f1=ones(n,1);
f=Y*f1;
A=[-Y';Y'];
b=[zeros(n,1);ones(n,1)];
Aeq=t';
Beq=1;
w= linprog(f,A,b,Aeq,Beq);
y=w'*Y;
end

