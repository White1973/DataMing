%读取数据
X=load('data.txt');
%求均值矩阵
meanX=mean(X);
%求标准差
stdX=std(X);
[M,N]=size(X);
%去除平均值
X1=(X-repmat(meanX,M,1))./repmat(stdX,M,1);
%X1=X-repmat(meanX,M,1);
%求协方差
covariance=cov(X1);
%求特征值和特征向量
[eigV,eigD]=eig(covariance);
%特征值从大到小排序
eigD=diag(eigD);
[junk,rindices]=sort(-1*eigD);%junk表示eigD逆序同时值*-1,rindices为n,n-1...1
eigD=eigD(rindices);
%特征向量按照特征值的方式进行排序
eigV=eigV(:,rindices);
%投影
V=eigV(:,1);
result=X*V;


