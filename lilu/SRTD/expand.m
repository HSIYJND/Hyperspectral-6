function XExpand=expand(X,winlenth)
SizeA=size(X);
XExpand((winlenth+1)/2:(winlenth-1)/2+SizeA(1), (winlenth+1)/2:(winlenth-1)/2+SizeA(2),:)=X; %中
XExpand((winlenth-1)/2:-1:1,(winlenth+1)/2:(winlenth-1)/2+SizeA(2),:)=X(1:(winlenth-1)/2,:,:); %上
XExpand(winlenth-1+SizeA(1):-1:(winlenth+1)/2+SizeA(1),(winlenth+1)/2:(winlenth-1)/2+SizeA(2),:)...%下
=X(SizeA(1)-(winlenth-3)/2:SizeA(1),:,:); %
XExpand((winlenth+1)/2:(winlenth-1)/2+SizeA(1),(winlenth-1)/2:-1:1,:)=X(:,1:(winlenth-1)/2,:); %左
XExpand((winlenth+1)/2:(winlenth-1)/2+SizeA(1),winlenth-1+SizeA(2):-1:(winlenth+1)/2+SizeA(2),:)...%右
=X(:,SizeA(2)-(winlenth-3)/2:SizeA(2),:); %
XExpand((winlenth-1)/2:-1:1,(winlenth-1)/2:-1:1,:)=X(1:(winlenth-1)/2,1:(winlenth-1)/2,:); %左上
XExpand(winlenth-1+SizeA(1):-1:(winlenth+1)/2+SizeA(1),winlenth-1+SizeA(2):-1:(winlenth+1)/2+... %右下
SizeA(2),:)=X(SizeA(1)-(winlenth-3)/2:SizeA(1),SizeA(2)-(winlenth-3)/2:SizeA(2),:); %
XExpand(winlenth-1+SizeA(1):-1:(winlenth+1)/2+SizeA(1),(winlenth-1)/2:-1:1,:)... %左下
=X(SizeA(1)-(winlenth-3)/2:SizeA(1),1:(winlenth-1)/2,:); %
XExpand((winlenth-1)/2:-1:1,winlenth-1+SizeA(2):-1:(winlenth+1)/2+SizeA(2),:)... %右上
=X(1:(winlenth-1)/2,SizeA(2)-(winlenth-3)/2:SizeA(2),:); %