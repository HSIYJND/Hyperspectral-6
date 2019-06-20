function [B,D,x]=extract(Y,m,n,winlenth,SizeI)

a=Y(m,n,:);
[~,~,b]=size(a);
for i=1:b,
    x(i,1)=a(1,1,i);
end
B=Y(((m-(winlenth-1)/2):(m+(winlenth-1)/2)),((n-(winlenth-1)/2):...
    (n+(winlenth-1)/2)),:);
B((winlenth+1)/2-(SizeI-1)/2:(winlenth+1)/2+(SizeI-1)/2,(winlenth+1)/2-...
    (SizeI-1)/2:(winlenth+1)/2+(SizeI-1)/2,:)=0;

k=0;
for i=1:winlenth,
    for j=1:winlenth,
        k=k+1;
        BB(:,k)=B(i,j,:);
    end
end

k=1;
for i=1:winlenth*winlenth,
    if BB(:,i)==0
        continue;
    else
        D(:,k)=BB(:,i);
        k=k+1;
    end
end

D = normalize_data(D);
    