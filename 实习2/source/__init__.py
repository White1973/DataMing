#-*- coding:utf-8 -*-
import numpy as np;
import method1.bayes as by ;
import random as random;
def spamFilter():
    classList=[];fullData=[];docData=[];
    for i in range(1,26):
        f=open("email/ham/"+str(i)+".txt","r");         #好的邮件
        data=by.textParge(f.read());
        classList.append(0);
        fullData.extend(data);
        docData.append(data);

        f=open("email/spam/"+str(i)+".txt","r")           #垃圾邮件
        data = by.textParge(f.read());
        classList.append(1);
        fullData.extend(data);
        docData.append(data);

    vocabList = by.createVocabList(docData);  #生成词汇表

    #测试集生成
    traingSet=list(range(50));testSet=[];testClass=[];
    for i in range(10):
        #randomIndex=int(random.uniform( 0,len(traingSet)));      #随机数生成包含上下限
        randomIndex=random.randint(0,len(traingSet)-1)
        testSet.append(traingSet[randomIndex]);            #traingSet[randomIndex]与randomIndex的区别
        testClass.append(classList[randomIndex]);
        del(traingSet[randomIndex]);                       #

    #训练集训练提取条件概率（训练集生成）
    trainMat=[];
    trainClass=[];
    setTrainMat=[];
    for item in traingSet:
        trainMat.append(by.setWordToVec(vocabList,docData[item]));
        trainClass.append(classList[item]);
        setTrainMat.append(by.SetsetWordToVec(vocabList,docData[item]));

    proList0,proList1,probolity=by.trainNB0(trainMat,trainClass);         #计算个字的条件概率
    setproList0,setproList1,setprobolity=by.trainNB0(setTrainMat,trainClass);
    '''
    #训练集训练并推断出错误率
    errorCount=0;setErrorCount=0;
    for item in testSet:
        docVec=by.setWordToVec(vocabList,docData[item]);
        #print(docVec)
        #print(by.classifyNB(docVec, proList0, proList1, probolity), classList[item])
        if by.classifyNB(docVec,proList0,proList1,probolity)!=classList[item]:
            #print(by.classifyNB(docVec,proList0,proList1,probolity),classList[item])
            errorCount+=1;
        setDocVec=by.SetsetWordToVec(vocabList,docData[item]);   #词集方法计算邮件过滤的错误率
        if by.classifyNB(setDocVec,setproList0,setproList1,setprobolity)!=classList[item]:
            setErrorCount+=1;
    '''
    TP=0;FP=0;TN=0;FN=0;
    setTP=0;setFP=0;setFN=0;setTN=0;
    for item in testSet:
        docVec=by.setWordToVec(vocabList,docData[item]);
        #print(docVec)
        #print(by.classifyNB(docVec, proList0, proList1, probolity), classList[item])
        if by.classifyNB(docVec,proList0,proList1,probolity)==0 and classList[item]==0:
            #print(by.classifyNB(docVec,proList0,proList1,probolity),classList[item])
            TP+=1;
        elif by.classifyNB(docVec,proList0,proList1,probolity)==0 and classList[item]==1:
            #print(by.classifyNB(docVec,proList0,proList1,probolity),classList[item])
            FP+=1;
        elif by.classifyNB(docVec,proList0,proList1,probolity)==1 and classList[item]==0:
            #print(by.classifyNB(docVec,proList0,proList1,probolity),classList[item])
            FN+=1;
        elif by.classifyNB(docVec, proList0, proList1, probolity) == 1 and classList[item] == 1:
            # print(by.classifyNB(docVec,proList0,proList1,probolity),classList[item])
            TN += 1;
        setDocVec=by.SetsetWordToVec(vocabList,docData[item]);   #词集方法计算邮件过滤的错误率
        if by.classifyNB(setDocVec,setproList0,setproList1,setprobolity)==0 and classList[item]==0:
            setTP+=1;
        elif by.classifyNB(setDocVec,setproList0,setproList1,setprobolity)==1 and classList[item]==0:
            setFP+=1;
        elif by.classifyNB(setDocVec,setproList0,setproList1,setprobolity)==0 and classList[item]==1:
            setFN+=1;
        elif by.classifyNB(setDocVec,setproList0,setproList1,setprobolity)==1 and classList[item]==1:
            setTN+=1;

    #return float(errorCount)/float(len(testSet)),float(setErrorCount)/float(len(testSet));
    return float(TP)/float(TP+FP),float(TP)/float(FP+TP),float(TP+TN)/float(len(testSet)), \
           float(setTP) / float(setTP + setFP), float(setTP) / float(setFP + setTP), float(setTP + setTN) / float(len(testSet));

if __name__=="__main__":
    #f = open("email/ham/23.txt", "r",encoding="gbk");  # 好的邮件
    #data = by.textParge(f.read());
    #print(data);
    #print(spamFilter());
    '''
    rates=[];setRates=[];
    for i in range(1000):
      rate,setRate=spamFilter();
      rates.append(rate);
      setRates.append(setRate);

    print("词袋法邮件过滤的错误率:" + str(sum(rates)/len(rates)));
    print("词集法邮件过滤的错误率:" + str(sum(setRates)/len(setRates)));
    '''
    accurates=[]  #精确率
    precises=[];  #准确率
    recallrates=[]  #召回率
    #词集分析
    setAccurates = []  # 精确率
    setPrecises = [];  # 准确率
    setRecallrates = []  # 召回率
    #rates = [];
    #setRates = [];
    for i in range(100):
        rate1, rate2,rate3,setrate1,setrate2,setrate3 = spamFilter();
        accurates.append(rate1)
        precises.append(rate3);
        recallrates.append(rate2);

        setAccurates.append(setrate1);
        setRecallrates.append(setrate2);
        setPrecises.append(setrate3);


    print("词袋法邮件过滤的精确率:" + str(sum(accurates) / len(accurates)));
    print("词袋法邮件过滤的召回率:" + str(sum(recallrates) / len(recallrates)));
    print("词袋法邮件过滤的准确率:" + str(sum(precises) / len(precises)));

    print("-------------------------------------------")
    
    print("词集法邮件过滤的精确率:" + str(sum(setAccurates) / len(setAccurates)));
    print("词集法邮件过滤的召回率:" + str(sum(setRecallrates) / len(setRecallrates)));
    print("词集法邮件过滤的准确率:" + str(sum(setPrecises) / len(setPrecises)));