#-*- coding:utf-8 -*-
import xlrd
import numpy as np;
import matplotlib.image as mpimg
from skimage import transform #压缩图像
from skimage import feature as skft   #LBP特征提取
from sklearn.naive_bayes import GaussianNB;
from sklearn.svm import  SVR
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
import random as random;
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
import cv2
import emotionJudge.NeuralNet as NN;
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold;

def LoadData():
    dataSet=[];dataLabel=[];
    info=xlrd.open_workbook("URLsWithEmotionCat.xlsx");
    info_sheet1=info.sheet_by_index(0);
    #print(info_sheet1.cell_value(1,0));
    #print(info_sheet1);
    rows,columns=info_sheet1.nrows,info_sheet1.ncols;
    #count=0;
    #print(rows,columns);
    for i in range(1,rows):
        #根据xlsx中第一列的数据获取图像名称,为字符串中最后一个/后面的字符串
        url=info_sheet1.cell_value(i,0);
        temp=re.split('/',url);
        imageName=temp[len(temp)-1];

        #读取图像
        try:
            #获取初始图像
           init_image=mpimg.imread("urlf/"+imageName);
           #缩放减少计算量
           image=transform.resize(init_image,(200,200));
           #灰度化
           gravity = np.array([0.299, 0.587, 0.114])
           data=np.array(image);
           data=np.dot(data,gravity.T)
           #将其转为一维数组
           #data=data.flatten()

           for j in range(1,columns):
               if(int(info_sheet1.cell_value(i,j))!=-1):
                   dataLabel.append(j-1);
                   break;

           dataSet.append(data);

           #判断图像是否缩放成功
           #plt.figure('resize')
           #plt.subplot(111)
           #plt.title('before resize')
           #plt.imshow(image, plt.cm.gray)
           #picture.reshape(48,48);
           #print("he")
        except:
            continue;
            print(i,imageName);
            print("This picture isn't exist");
    print(len(dataSet));

    np.save("docData.npy",dataSet);
    np.save("docLabel.npy",dataLabel);

    dataSet=np.load("docData.npy");
    print(np.array(dataSet).shape)
    dataLabel=np.load("docLabel.npy");
    print(np.array(dataLabel).shape)
    return dataSet,dataLabel;

        #print(image);

#获取训练集和测试集
def TrainAndTest(dataSet,dataLabel):
    #print(len(dataSet));
    testMat=[];testLabel=[];trainMat=[];trainLabel=[];
    l=[random.uniform(0,1) for i in range(len(dataSet))];
    #print(len(l));
    #生成测试集,训练集
    for i in range(len(dataSet)):
        #小于0.8的归为训练集
        if l[i]<0.8:
            trainMat.append(dataSet[i]);
            trainLabel.append((dataLabel[i]));
        #大于等于0.8归为测试集
        else:
            testMat.append(dataSet[i]);
            testLabel.append(dataLabel[i]);

    #print(len(trainMat),len(testMat));
    return trainMat,trainLabel,testMat,testLabel;

radius=1;
points=8;
def LBP_doc(docMat):
    hist_doc=np.zeros((len(docMat),256));
    #np.save("LBPData.npy",hist_doc);
    for i in range(len(docMat)):
        #print(np.array(trainMat[i]).shape)
        lbp=skft.local_binary_pattern(docMat[i],points,radius,"default");
        #print(np.array(lbp).shape);
        max_bin=int(lbp.max()+1);
        hist_doc[i],_=np.histogram(lbp,normed=True,bins=max_bin,range=(0,max_bin));
    return hist_doc;


def LBP(trainMat,testMat):
    print(len(trainMat));
    print(len(testMat));
    hist_train=np.zeros((len(trainMat),256));
    hist_test=np.zeros((len(testMat),256));
    for i in range(len(trainMat)):
        #print(np.array(trainMat[i]).shape)
        lbp=skft.local_binary_pattern(trainMat[i],points,radius,"default");
        #print(np.array(lbp).shape);
        max_bin=int(lbp.max()+1);
        hist_train[i],_=np.histogram(lbp,normed=True,bins=max_bin,range=(0,max_bin));

    for i in range(len(testMat)):
        lbp=skft.local_binary_pattern(testMat[i],points,radius,"default");
        #print(np.array(lbp).shape);
        max_bin=int(lbp.max()+1);
        hist_test[i],_=np.histogram(lbp,normed=True,bins=max_bin,range=(0,max_bin));

    return hist_train,hist_test;

def HOG(trainMat,testMat):
     normalised_blocks, hog_image = skft.hog(trainMat[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualise=True);

     print(np.array(hog_image).shape);
     print(normalised_blocks);
     print(np.array(normalised_blocks).shape);
     cv2.imshow("pic.jpg", hog_image);



def pca(arrayMat,percentage):
    pca = PCA(n_components=percentage);  # ;读取arrayMat时'mle'的情况不行
    n_mat=pca.fit_transform(arrayMat);
    ratio=pca.explained_variance_ratio_;
    return n_mat;


def SIFT():
    dataSet = [];
    dataLabel = [];

    info = xlrd.open_workbook("URLsWithEmotionCat.xlsx");
    info_sheet1 = info.sheet_by_index(0);

    rows, columns = info_sheet1.nrows, info_sheet1.ncols;
    Array_des=[];
    for i in range(1, rows):
        # 根据xlsx中第一列的数据获取图像名称,为字符串中最后一个/后面的字符串
        url = info_sheet1.cell_value(i, 0);
        temp = re.split('/', url);
        imageName = temp[len(temp) - 1];

        # 读取图像
        try:
            # 获取初始图像
            init_image = cv2.imread("urlf/" + imageName);
            # 缩放减少计算量
            #image = transform.resize(init_image, (200, 200));
            # 灰度化
            gray=cv2.cvtColor(init_image,cv2.COLOR_RGB2GRAY);
            #siftDetector=cv2.xfeatures2d.SIFT_create(100);
            siftDetector =cv2.xfeatures2d.SIFT_create(23);
            kp,des=siftDetector.detectAndCompute(gray, None);
            #des=des[:24,:];
            new_des=pca(des,15);
            new_des=new_des[:23,:];

            print(np.array(new_des).shape)
            new_des=np.array(new_des).flatten();
            #Array_des.append(new_des)
            # gravity = np.array([0.299, 0.587, 0.114])
            # data = np.array(image);
            # data = np.dot(data, gravity.T)
            # 将其转为一维数组
            # data=data.flatten()

            for j in range(1, columns):
                if (int(info_sheet1.cell_value(i, j)) != -1):
                    dataLabel.append(j - 1);
                    break;
            Array_des.append(new_des)
            #dataSet.append(data);

            # 判断图像是否缩放成功
            # plt.figure('resize')
            # plt.subplot(111)
            # plt.title('before resize')
            # plt.imshow(image, plt.cm.gray)
            # picture.reshape(48,48);
            # print("he")
        except:

            continue;
            # print(i,imageName);
            # print("This picture isn't exist");
    print(len(Array_des));
    #Array_des=np.load("SIFT.npy");
    np.save("docLabel.npy",dataLabel);
    dataLabel=np.load("docLabel.npy");
    print(len(dataLabel));
    #Array_des=Array_des.reshape(Array_des.shape[0],-1);
    np.save("SIFT.npy",Array_des);
    #Array_des=np.array(Array_des);
    #np.save("SIFT.npy",Array_des);
    print(np.array(Array_des).shape);
    #print(min(Array))
    return Array_des,dataLabel;


#根据testLabel和predict来制作混淆矩阵
def my_confusion_matrix(testLabel,predict):
    labels=list(set(testLabel));     #获得类别数
    con_mat=confusion_matrix(testLabel,predict);
    print(type(con_mat))
    print(len(con_mat),len(con_mat[0]))
    print("confusion_matrix:");
    print("label:\t",end='');
    for i in range(len(labels)):
        print(str(labels[i])+"\t",end='');
    print();
    for i in range(len(labels)):
        print(str(i)+"\t\t",end='');
        for j in range(len(con_mat)):
            print(str(con_mat[i][j])+"\t",end='');
        print();
    print();

    #准确率计算
    true_num=0;  #预测正确的个数
    for i in range(len(con_mat)):
        true_num+=con_mat[i,i];
    Arrucate=float(true_num)/float(np.array(con_mat).sum());
    print("准确率:"+str(Arrucate));
    #召回率计算，所有值的召回率求平均
    Recall=0.0;
    for i in range(len(con_mat)):
        #print(con_mat[:,i])
        if sum(con_mat[:,i])!=0:
           Recall+=float(con_mat[i][i])/float(sum(con_mat[:,i]));

    Recall=Recall/float(len(con_mat[0]));
    print("召回率:"+str(Recall));

    #求精确率，所有精确率求平均
    Precise=0;
    for i in range(len(con_mat)):
        #print(con_mat[i,:])
        Precise+=float(con_mat[i][i])/float(sum(con_mat[i,:]));
    Precise=Precise/float(len(con_mat[0]));
    print("精确率:"+str(Precise))

    return Arrucate;

def SVM_Method(trainData,trainLabel,testData,testLabel):
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1);
    #result = OneVsRestClassifier(svr_rbf, -1).fit(trainData, trainLabel).score(testData, testLabel)
    result = OneVsRestClassifier(svr_rbf, -1).fit(trainData, trainLabel).predict(testData);
    return result;

def DTC_Method(trainData,trainLabel,testData,testLabel):
    dtc = DecisionTreeClassifier();
    dtc.fit(trainData,trainLabel);
    #result=dtc.score(testData,testLabel);
    result=dtc.predict(testData);
    return result;

def Bayes_Method(trainData,trainLabel,testData,testLabel):
    gl=GaussianNB();
    gl.fit(trainData,trainLabel);
    #result=gl.score(testData,testLabel);
    result=gl.predict(testData);
    return result;


def NeuralNet_Method(trainData,trainLabel,testData,testLabel):
    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
                        max_iter=500, verbose=10, learning_rate_init=.1)
    mlp.fit(trainData,trainLabel);
    #result=mlp.score(testData,testLabel);
    result=mlp.predict(testData);

    #result=NN.NueralNet(trainData,trainLabel,testData,testLabel);
    return result;

#k折交叉验证
#def K_Cross_Validation(docData,docLabel,k):
def K_Cross_Validation(k):
    docLabel=np.load("docLabel.npy");
    print(len(docLabel));
    docLabel = np.array(docLabel);
    kf=KFold(n_splits=k);
    Accurate_NN=0.0;Accurate_SVM=0.0;Accurate_DTC=0.0;Accurate_Bayes=0.0;
    #hist_data=np.array(LBP_doc(docData));
    #np.save("LBPData.npy",hist_data);

    hist_data=np.load("LBPData.npy");
    #hist_data=np.load("SIFT.npy");
    print(np.array(hist_data).shape)
    for trainIdex,testIndex in kf.split(hist_data):

        #hist_data,docLabel必须为np.array型
        trainData,testData=hist_data[trainIdex],hist_data[testIndex];
        trainLabel,testLabel=docLabel[trainIdex],docLabel[testIndex];

        #神经网络
        print("NeuralNet_Method:")
        predict=NeuralNet_Method(trainData,trainLabel,testData,testLabel);
        Ac=my_confusion_matrix(testLabel,predict);
        Accurate_NN+=Ac;

        #SVM
        print("SVM_Method:");
        predict=SVM_Method(trainData,trainLabel,testData,testLabel);
        Ac=my_confusion_matrix(testLabel,predict);
        Accurate_SVM+=Ac;
        #决策树
        print("DTC_Method:");
        predict=DTC_Method(trainData,trainLabel,testData,testLabel);
        Ac=my_confusion_matrix(testLabel,predict);
        Accurate_DTC+=Ac;

        #贝叶斯方法
        print("Bayes_Method");
        predict=Bayes_Method(trainData,trainLabel,testData,testLabel);
        Ac=my_confusion_matrix(testLabel,predict)
        Accurate_Bayes+=Ac;
        #n_jobs是多线程,-1代表全部cpu资源都用上
        # result = OneVsRestClassifier(dtc, -1).fit(trainData, trainLabel).score(testData, testLabel)

        # dtc = DecisionTreeClassifier();
        # dtc.fit(trainData,trainLabel);
        # result=dtc.score(testData,testLabel);

        #Accurate+=result;
    print("NeuralNet_Method:"+str(float(Accurate_NN/float(k))));
    print("SVM_Method:" + str(float(Accurate_SVM / float(k))));
    print("DTC_Method:" + str(float(Accurate_DTC / float(k))));
    print("Bayes_Method:" + str(float(Accurate_Bayes / float(k))));
    #return float(Accurate_NN/float(k));

if __name__=="__main__":
    K_Cross_Validation(5)
    #dataSet,dataLabel=LoadData();

    #print(K_Cross_Validation(dataSet,dataLabel,5));
    #print(K_Cross_Validation(5))
    #trainMat, trainLabel, testMat, testLabel=TrainAndTest(dataSet,dataLabel);
   # #  print(np.array(dataLabel).min(),np.array(dataLabel).max())
   # #  # print("trining...................")
   # #  # print(trainMat,trainLabel);
   # #  # print("testing...................")
   # #  # print(testMat,testLabel);
   # #
    #HOG(trainMat,testMat);
    #hist_train,hist_test=LBP(trainMat,testMat);
    #SIFT(trainMat,testMat);
    #SIFT();
   # #  print("After LBP..........................");
   # #  print(hist_train,hist_test);
   # #
   #
   #  print("SVM result:");
   #  cls=svm.LinearSVC();
   #  cls.fit(hist_train,trainLabel);
   #
   #  predict=cls.predict(hist_test);
   #  '''
   #
   #  print("DecisionTreeClassifier result:");
   #  dtc=DecisionTreeClassifier();
   #  dtc.fit(hist_train,trainLabel);
   #  result=dtc.score(hist_test,testLabel);
   #  print(result);
    #predict=dtc.predict(hist_test,testLabel);
   #  '''
   #  my_confusion_matrix(testLabel,predict);
   #  #print("Neural result:");
   #  #result=NN.NueralNet(np.array(trainMat),trainLabel,np.array(testMat),testLabel);
   #  #print("result:",result);
   #  '''
   #  count=0;
   #  gl=GaussianNB();
   #  gl.fit(hist_train,trainLabel);
   #  #preduct_gl=gl.predict(hist_test);
   #  result=gl.score(hist_test,testLabel);
   #  print(result)
   #  for i in range(len(hist_test)):
   #      if testLabel[i]==preduct_gl[i]:
   #          count+=1;
   # #  '''
   #  print("result:"+str(float(count)/float(len(preduct_gl))));
   #
   #
   # my_confusion_matrix(testLabel,preduct_gl);

