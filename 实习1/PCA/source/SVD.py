import numpy as np;

import scipy.io as scipy;
import matplotlib.pyplot as plt;
import matplotlib.colors as colors;
def readFile():
    file="data.mat";
    Data=scipy.loadmat(file);
    Data=np.array(Data['data']);

    column=Data.shape[1]-1;
    arrayMat=Data[:,0:column];
    label=Data[:,column]
    return np.mat(arrayMat),label;

#将奇异值组成的向量转化为奇异值矩阵
def sigmalVectorToMatrix(S):
    retMat=np.eye(len(S));
    for i in range(len(S)):
        retMat[i]*=S[i];
    return retMat;
#当前计算SVD要提取的奇异值的数量
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

def paint(n_mat, label):
    n_mat = np.mat(n_mat);
    fig = plt.figure();
    color = colors.ListedColormap(['r', 'g', 'b', 'c', 'y', 'k']);
    ax = fig.add_subplot(111);

    ax.scatter(n_mat[:, 0].flatten().A[0], n_mat[:, 1].flatten().A[0], marker='o', s=30, c=label, cmap=color);
    # plt.legend();

    plt.show();



#SVD降维
def SVDepress(mat,percentage):
    U,S,VT=np.linalg.svd(mat);
    print(type(U),type(VT));
    N=percentageCal(S,percentage);
    print(S);print(N);
    U=U[:,:N];
    VT=VT[:N];
    S=S[:N];
    sigmal=sigmalVectorToMatrix(S);
    SimiMat=U*sigmal*VT;
    return SimiMat;



if __name__=="__main__":
    list1=([[1,2,3],
           [2,3,4],
           [4,5,6]]);

    mat,label=readFile();
    recont=SVDepress(mat,0.93);
    paint(recont,label)
    print(recont);


