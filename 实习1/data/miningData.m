%��ȡ����
X=load('data.txt');
%���ֵ����
meanX=mean(X);
%���׼��
stdX=std(X);
[M,N]=size(X);
%ȥ��ƽ��ֵ
X1=(X-repmat(meanX,M,1))./repmat(stdX,M,1);
%X1=X-repmat(meanX,M,1);
%��Э����
covariance=cov(X1);
%������ֵ����������
[eigV,eigD]=eig(covariance);
%����ֵ�Ӵ�С����
eigD=diag(eigD);
[junk,rindices]=sort(-1*eigD);%junk��ʾeigD����ͬʱֵ*-1,rindicesΪn,n-1...1
eigD=eigD(rindices);
%����������������ֵ�ķ�ʽ��������
eigV=eigV(:,rindices);
%ͶӰ
V=eigV(:,1);
result=X*V;


