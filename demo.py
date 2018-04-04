import numpy as np
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
def paint(n_mat,labels):
    n_mat=np.mat(n_mat);
    fig=plt.figure();
    #color=colors.ListedColormap(['r','g','b','c','y','k']);
    color=['r','g','b','c','y','k'];

    ax=fig.add_subplot(111);
    for i in range(6):
       vects=n_mat[labels==i]
       #ax.scatter(vects[:, 0].flatten().A[0], vects[:, 1].flatten().A[0], label=i, marker='o', s=30, c=label[i],cmap=color[i]);
       ax.scatter(vects[:, 0].flatten().A[0], vects[:, 1].flatten().A[0], label=i, marker='o', s=30, c=color[i]);
       plt.legend();
    plt.show();

if __name__=="__main__":
    mat,label=readFile();
    matVects,ratio=LDA(mat,label);
    n_mat=SneDepress(matVects);
    paint(n_mat,label)

'''
a=np.array([[1,2,3],
   [2,3,4],
   [36,4,5]])
b=np.array([1,1,2]);
a=a[b==2];
print(a);
'''