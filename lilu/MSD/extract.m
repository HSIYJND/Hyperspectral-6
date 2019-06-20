function [V_b,V_t]=extract(Y,t)

y=COV(Y);
[V,D]=eig(y);
V=fliplr(V);
V_b=V(:,1:3);

x=COV(t);
[V,D]=eig(x);
V=fliplr(V);
V_t=V(:,1:6);
    
   

    