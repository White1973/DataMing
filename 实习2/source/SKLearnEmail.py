from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np;
import random as random;
import method1.bayes as by;
def TestCase():
    classList = [];
    fullData = [];
    docData = [];
    for i in range(1, 26):
        f = open("email/ham/" + str(i) + ".txt", "r");  # 好的邮件
        data = by.textParge(f.read());
        classList.append(0);
        fullData.extend(data);
        docData.append(data);

        f = open("email/spam/" + str(i) + ".txt", "r")  # 垃圾邮件
        data = by.textParge(f.read());
        classList.append(1);
        fullData.extend(data);
        docData.append(data);

    vocabList = by.createVocabList(docData);  # 生成词汇表

    #测试集，训练集生成
    traingSet=list(range(50));testSet=[];testClass=[];
    for i in range(10):
        #randomIndex=int(random.uniform( 0,len(traingSet)));      #随机数生成包含上下限 均匀分布
        randomIndex=random.randint(0,len(traingSet)-1)
        testSet.append(traingSet[randomIndex]);            #?????????????traingSet[randomIndex]与randomIndex的区别
        testClass.append(classList[randomIndex]);
        del(traingSet[randomIndex]);                       #????????????????


    #训练集训练提取条件概率
    trainMat=[];
    trainClass=[];
    setTrainMat=[];
    for item in traingSet:
        trainMat.append(by.setWordToVec(vocabList,docData[item]));
        trainClass.append(classList[item]);
        setTrainMat.append(by.SetsetWordToVec(vocabList,docData[item]));

    glf=BernoulliNB();
    glf.fit(trainMat,trainClass);
    error=0;
    for item in testSet:
       docVec=[by.setWordToVec(vocabList,docData[item])];
       result=glf.predict(docVec);
       if result[0]!=classList[item]:
           error+=1;

    setError=0;
    glf = BernoulliNB();
    glf.fit(setTrainMat, trainClass);
    for item in testSet:
        setDocVec=[by.SetsetWordToVec(vocabList,docData[item])];
        result=glf.predict(setDocVec);
        print(result[0],classList[i])
        if result[0]!=classList[i]:
            setError+=1;

    return float(error)/float(len(testSet)),float(setError)/float(len(testSet));
    #result=glf.predict(testSet);
    #print(result);

if __name__=="__main__":
    rates=[];setRates=[];
    for i in range(10):
      rate,setRate=TestCase();
      rates.append(rate);
      setRates.append(setRate);
      print("----------------------")
      print(rate,setRate)


    print("词集统计错误率:"+str(sum(setRates)/len(setRates)));
    print("词袋错误率统计"+str(sum(rates)/len(rates)));

