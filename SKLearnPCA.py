from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.io as scipy
import numpy as np;
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def readFile():
    file="data.mat";
    Data=scipy.loadmat(file);
    Data=np.array(Data['data']);

    column=Data.shape[1]-1;
    arrayMat=Data[:,0:column];
    label=Data[:,column]
    return arrayMat,label;


def pca(arrayMat,percentage):
    pca = PCA(n_components=percentage);  # ;读取arrayMat时'mle'的情况不行
    n_mat=pca.fit_transform(arrayMat);
    ratio=pca.explained_variance_ratio_;

    return n_mat,ratio;

#归一化过程
def featureNormalize(X):
    '''（每一个数据-当前列的均值）/当前列的标准差'''
    n = X.shape[1]
    l=X.shape[0];
    meanVal = np.zeros((1, n));
    sigma = np.zeros((1, n))

    meanVal = np.mean(X, axis=0)  # axis=0表示列
    sigma=np.std(X,axis=0)
    newData=X-meanVal;
    return newData,meanVal;

#当前计算PCA要提取的特征值的数量
def percentageCal(eigVals,percent):
    arraySort=np.sort(eigVals);
    arraySort=arraySort[-1::-1];     #倒叙

    sumVects=sum(eigVals);
    num=0;
    total=0;
    for vect in eigVals:
        num+=1;
        total+=vect;
        if total>=sumVects*percent:
            return num;

#PCA降维过程
def PCADepress(mat,percent):
    newData, meanVal = featureNormalize(mat);
    covMat = np.cov(newData, rowvar=0);  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat));  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n = percentageCal(eigVals, percent);  # 获取要选择的特征值数n
    eigValIndice = np.argsort(eigVals);  # 对特征值从小到大排序，返回次序
    n_eigVals = eigValIndice[-1:-(n + 1):-1];

    n_eigVect = eigVects[:, n_eigVals]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    # 最后输出的低维矩阵
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据

    return lowDDataMat, reconMat;

def PCA(mat, N):
        newData, meanVal = featureNormalize(mat);
        covMat = np.cov(newData, rowvar=0);  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

        eigVals, eigVects = np.linalg.eig(np.mat(covMat));  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        #n = percentageCal(eigVals, percent);  # 获取要选择的特征值数n
        eigValIndice = np.argsort(eigVals);  # 对特征值从小到大排序，返回次序
        n_eigVals = eigValIndice[-1:-(N + 1):-1];

        n_eigVect = eigVects[:, n_eigVals]  # 最大的n个特征值对应的特征向量
        lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
        # 最后输出的低维矩阵
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据

        return lowDDataMat, reconMat;
    #pass;
#可视化过程
def paint(n_mat,label):
    n_mat=np.mat(n_mat);
    fig=plt.figure();
    color=colors.ListedColormap(['r','g','b','c','y','k']);
    ax=fig.add_subplot(111);
    ax.scatter(n_mat[:, 0].flatten().A[0], n_mat[:, 1].flatten().A[0], marker='o', s=30, c=label,cmap=color);
    plt.show();

#T-SNE降维方法
def SneDepress(X_mat):
    tsne=TSNE(n_components=2,init="pca",random_state=0);
    X_tsne=tsne.fit_transform(X_mat);

    return X_tsne;

if __name__=="__main__":
      arrayMat,label=readFile();
      #print(arrayMat);

      #print(label);

      n_mat,ratio=pca(arrayMat,2 );
      #n_mat,recont=PCA(arrayMat,2);

      #low_mat,n_mat=PCADepress(arrayMat,0.85);
      #new_mat=SneDepress(n_mat)
      paint(n_mat,label);
      ##n_arrayMat,ratio=pca(arrayMat,2);

#arrayMat = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
      #pca = PCA(n_components=2); #;读取arrayMat时'mle'的情况不行
      #pca.fit(arrayMat);

#print(pca.explained_variance_ratio_)
#print(sum(pca.explained_variance_ratio_));
#print(pca.n_components)


