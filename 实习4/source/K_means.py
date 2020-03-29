#-*- coding:utf-8 -*-
import xlrd
import numpy as np;
import matplotlib.image as mpimg
from skimage import transform #压缩图像
from skimage import feature as skft   #LBP特征提取
import random as random;
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import re
import cv2
from sklearn.model_selection import KFold;
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
COLUMN=16;
#读取图像数据
def LoadData():
    # dataSet=[];dataLabel=[];
    # info=xlrd.open_workbook("URLsWithEmotionCat.xlsx");
    # info_sheet1=info.sheet_by_index(0);
    # rows,columns=info_sheet1.nrows,info_sheet1.ncols;
    #
    # for i in range(1,rows):
    #     #根据xlsx中第一列的数据获取图像名称,为字符串中最后一个/后面的字符串
    #     url=info_sheet1.cell_value(i,0);
    #     temp=re.split('/',url);
    #     imageName=temp[len(temp)-1];
    #
    #     #读取图像
    #     try:
    #         #获取初始图像
    #        init_image=mpimg.imread("urlf/"+imageName);
    #        #缩放减少计算量
    #        image=transform.resize(init_image,(200,200));
    #        #灰度化
    #        gravity = np.array([0.299, 0.587, 0.114])
    #        data=np.array(image);
    #        data=np.dot(data,gravity.T)
    #
    #        for j in range(1,columns):
    #            if(int(info_sheet1.cell_value(i,j))!=-1):
    #                dataLabel.append(j-1);
    #                break;
    #        dataSet.append(data);
    #
    #     except:
    #         continue;
    #         print(i,imageName);
    #         print("This picture isn't exist");
    # print(len(dataSet));
    # np.save("docData.npy",dataSet);
    # np.save("docLabel.npy",dataLabel);
    dataSet=np.load("docData.npy");
    dataLabel=np.load("docLabel.npy");
    #print(np.array(dataSet).shape)

    return dataSet,dataLabel;


#LBP特征提取
radius=1;
points=8;
#def LBP_doc(docMat):
def LBP_doc():

    # hist_doc=np.zeros((len(docMat),256));
    # #np.save("LBPData.npy",hist_doc);
    # for i in range(len(docMat)):
    #     #print(np.array(trainMat[i]).shape)
    #     lbp=skft.local_binary_pattern(docMat[i],points,radius,"default");
    #     #print(np.array(lbp).shape);
    #     max_bin=int(lbp.max()+1);
    #     hist_doc[i],_=np.histogram(lbp,normed=True,bins=max_bin,range=(0,max_bin));

    #np.save("LBP_doc.npy",hist_doc);

    hist_doc=np.load("LBP_doc.npy")
    docLabel=np.load("docLabel.npy");
    return hist_doc,docLabel;

#求v1,v2的欧式距离
def eculid_distance(v1,v2):
    distance=np.sqrt(np.sum(np.power(v1-v2,2)));
    return distance;

#从数据集中选取k个质点
def randCent(dataset,k):
  try:
    n=np.shape(dataset)[1];   #确定数据集的维数
    centers=np.mat(np.zeros((k,n)));
    list=np.random.randint(0,len(dataset)-1,k);
    #print(list);
    centers=dataset[list];
    #print(np.array(centers).shape)
    # for i in range(n):
    #     mincol=min(dataset[:,i]);
    #     Range=float(max(dataset[:,i])-mincol);
    #     num=mincol+Range*np.random.rand(k,1);
    #     centers[:,i]=num;
    #     print(centers[:,i]);

    #print(centers);
    return centers;
  except:
      print(len(dataset));
      raise RuntimeError("Erroe")

#k_means聚类
def K_means(dataSet,K):
    n=np.shape(dataSet)[0];
    clussResult=np.mat(np.zeros((n,2)));     #分类结果统计  mat是为了每一项都是1*2的矩阵
    centers=randCent(dataSet,K);     #随机产生k个初始质点

    change=True;
    while change:
        change=False;
        for i in range(n):
            lowestdis=float("inf");minIndex=-1;
            for j in range(K):
                distance=eculid_distance(centers[j],dataSet[i]);
                if distance<lowestdis:
                    lowestdis,minIndex=distance,j;
                    #clussResult[i]=j,distance;
            if clussResult[i,0]!=minIndex:
                change = True;
            clussResult[i]=minIndex,lowestdis**2;                #lowestdis**2突出距离远的

        for k in range(K):
            points=dataSet[np.nonzero(clussResult[:,0]==k)[0]];
            centers[k]=np.mean(points,axis=0);                 #axis=0按列来取平均值

    #返回每个簇的中心点和每个点所属的类别
    return centers,clussResult;

#2分K-均值聚类
def binary_KMeans(dataSet,K):
    #初始为一个簇
    n = np.shape(dataSet)[0];
    clussResult = np.mat(np.zeros((n, 2)));  # 分类结果统计  mat是为了每一项都是1*2的矩阵
    center0=np.mean(dataSet,axis=0).tolist();
    centers=[center0];
    for i in range(n):
        clussResult[i,1]=(eculid_distance(center0,dataSet[i])**2);

    while len(centers)<K:
        lowestSSE=float("inf");
        print(len(centers));
        length=len(centers)
        #对每一个簇进行2分类，求出SSE
        for i in range(len(centers)):
            points=dataSet[np.nonzero(clussResult[:,0]==i)[0]];
            if len(points)<=1:continue;
            binary_centers,binary_clussResukt=K_means(points,2);
            binarySSE=np.sum(binary_clussResukt[:,1]);
            Non_binarySSE=np.sum(clussResult[np.nonzero(clussResult[:,0]!=i)[0],1]);

            #获取使整个数据集SSE最小的簇
            if (binarySSE+Non_binarySSE)<lowestSSE:
                bestIndex=i;
                lowestSSE=binarySSE+Non_binarySSE;
                best_bicenters=binary_centers;
                best_clussResult=binary_clussResukt;

        #2分类后分成best_clussResult[0]对应为0和1;
        #先计算==1，再计算==0，防止0，1转化为1，2时，0->1,则转化的1类和原来的1无法区分
        best_clussResult[np.nonzero(best_clussResult[:,0]==1)[0],0]=len(centers);
        #print(best_clussResult[np.nonzero(best_clussResult[:,0]==0)[0]]);
        #print(len(best_clussResult[np.nonzero(best_clussResult[:,0]==bestIndex)[0]]))
        best_clussResult[np.nonzero(best_clussResult[:,0]==0)[0],0]=bestIndex;
        #print(len(best_clussResult[np.nonzero(best_clussResult[:, 0] == len(centers))[0]]))
        centers[bestIndex]=best_bicenters[0];
        centers.append(best_bicenters[1]);
        clussResult[np.nonzero(clussResult[:,0]==bestIndex)[0],:]=best_clussResult;
        print("********************************************")
        for i in range(len(centers)):
            print(len(clussResult[np.nonzero(clussResult[:,0]==i)[0]]))

        print("*********************************************");
    return centers,clussResult;


def paint(n_mat,label,result):
    n_mat=np.mat(n_mat);
    fig=plt.figure();
    color=colors.ListedColormap(['r','g','b','c','y','k','m','orange','teal','violet','peru','crimson','tan','lime','gold','pink']);
    ax=fig.add_subplot(111);
    ax.scatter(n_mat[:, 0].flatten().A[0], n_mat[:, 1].flatten().A[0], marker='o', s=30, c=label,cmap=color);
    plt.title("Accurate Rate:"+str(result));
    plt.show();

def pca(arrayMat,percentage):
    pca = PCA(n_components=percentage);  # ;读取arrayMat时'mle'的情况不行
    n_mat=pca.fit_transform(arrayMat);
    ratio=pca.explained_variance_ratio_;

    return n_mat;
#T-SNE降维方法
def SneDepress(X_mat):
    tsne=TSNE(n_components=2,init="pca",random_state=0);
    X_tsne=tsne.fit_transform(X_mat);

    return X_tsne;

def Kmeas_SKLearn(dataset,K):
    estimator = KMeans(n_clusters=K,random_state=28)
    estimator.fit(dataset);
    #result=estimator.fit_predict(dataset);
    #result=estimator.score(dataset);
    label=estimator.labels_;
    centers=estimator.cluster_centers_;


    return label,centers;

#兰迪指数预测
def Kmeans_Evaluate(labels_true,labels_pred):
      result=metrics.adjusted_rand_score(labels_true,labels_pred);
      return result;

if __name__=="__main__":
    # docData,docLabel=LoadData();
    # print(np.shape(docData));
    # print(np.shape(docLabel));
    hist_doc,docLabel=LBP_doc();

    # print(np.array(hist_doc).shape);
    #
    #centers, clussResult = binary_KMeans(hist_doc, COLUMN);
    #centers,clussResult=K_means(hist_doc,COLUMN);
    # label, centers = Kmeas_SKLearn(hist_doc, COLUMN);
    #np.savez("K_means.npz",centers=centers,clussResult=clussResult);
    # Kmeans=np.load("K_means.npz");
    # clussResult=Kmeans["clussResult"];
    # centers=Kmeans["centers"];
    #label=np.array((clussResult[:,0])).flatten();
    #print(clussResult)

    #label,centers=Kmeas_SKLearn(hist_doc,COLUMN);
    #print(label);
    #print(np.array(label).shape);
    #print(np.array(docData).shape);
    #centers, clussResult = binary_KMeans(hist_doc, COLUMN);
    new_mat=SneDepress(hist_doc);
    #label, centers = Kmeas_SKLearn(new_mat, COLUMN);
    #centers, clussResult = K_means(new_mat, COLUMN);
    centers, clussResult = binary_KMeans(new_mat, COLUMN);
    label = np.array((clussResult[:, 0])).flatten();
    result=Kmeans_Evaluate(docLabel,label);
    print("result:"+str(result));

    #new_mat=pca(hist_doc,2);
    paint(new_mat,label,result);

   # print(np.array(centers).shape);

