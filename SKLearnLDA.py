import scipy.io as scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

def readFile():
    file="data.mat";
    Data=scipy.loadmat(file);
    Data=np.array(Data['data']);

    column=Data.shape[1]-1;
    arrayMat=Data[:,0:column];
    label=Data[:,column]
    return arrayMat,label;

def LDA(mat,label):
    lda=LinearDiscriminantAnalysis(n_components=5);
    lda.fit(mat,label);
    n_mat=lda.transform(mat);
    ratio=lda.explained_variance_ratio_;
    return n_mat,ratio;

#T-SNE降维方法
def SneDepress(X_mat):
    tsne=TSNE(n_components=2,init="pca",random_state=0);
    X_tsne=tsne.fit_transform(X_mat);

    return X_tsne;

#可视化过程
def paint(n_mat,label):
    n_mat=np.mat(n_mat);
    fig=plt.figure();
    color=colors.ListedColormap(['r','g','b','c','y','k']);
    ax=fig.add_subplot(111);
    #ax.scatter(n_mat[:, 0].flatten().A[0], n_mat[:, 1].flatten().A[0], marker='o', s=30, c=label, cmap=color);
    #for i in range(6):
    ax.scatter(n_mat[:, 0].flatten().A[0], n_mat[:, 1].flatten().A[0], marker='o', s=30, c=label,cmap=color);
    #plt.legend();

    plt.show();

if __name__=="__main__":
    mat,label=readFile();
    matVects,ratio=LDA(mat,label);
    n_mat=SneDepress(matVects);

    paint(n_mat,label);
