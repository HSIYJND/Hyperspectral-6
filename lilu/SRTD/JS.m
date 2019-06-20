function [c,d]=JS(S,Q)

a=S-Q;
b=Q-S;
c=find(a==1);
d=find(b==1);
c=length(c);
d=length(d);
