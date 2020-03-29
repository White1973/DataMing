import re
import numpy as ny;
import math as mt;

#切分文本
def textParge(bigString):
    listOfTokens=re.split(r'\W',bigString)       #以非字母数字的字符来划分
    #regEX=re.compile(r'\W*');
    #listOfTokekens=regEX.split(bigString);
    return [tok for tok in listOfTokens if len(tok)>2];

#创建词汇表
def createVocabList(dataSet):
    vocabSet=set([]);
    for item in dataSet:
        vocabSet=vocabSet|set(item);
    return list(vocabSet);

#词带统计inputSet在词汇表中词频
def setWordToVec(vocabList,inputSet):
    retVocab=[0]*len(vocabList);
    for item in inputSet:
        if item in vocabList:
            retVocab[vocabList.index(item)]+=1;
        else:
            print(item+"is not in vacabulary");
    return retVocab;

#词集统计inputSet在词汇表中词频
def SetsetWordToVec(vocabList,inputSet):
    retVocab=[0]*len(vocabList);
    for item in inputSet:
        if item in vocabList:
            retVocab[vocabList.index(item)]=1;
        else:
            print(item+"is not in vacabulary");
    return retVocab;

#计算训练集中各词汇的出现概率（分别在好的邮件和坏的邮件）条件概率
def trainNB0(trainMarix,trainCategory):
    line=len(trainMarix);
    column=len(trainMarix[0]);
    #print(sum(trainCategory),line)
    probality=sum(trainCategory)/float(line);          #训练集中垃圾邮件的概率
    pList0=ny.ones(column);pList1=ny.ones(column);
    pSum0=pSum1=float(column);
    for i in range(len(trainCategory)):
        if trainCategory[i]==0:
            pList0+=trainMarix[i];
            pSum0+=sum(trainMarix[i]);
        elif trainCategory[i]==1:
            pList1+=trainMarix[i];
            pSum1+=sum(trainMarix[i]);

    proList0=pList0/pSum0; proList1=pList1/pSum1;
    return proList0,proList1,probality;

#测试集代入模型分析过程
def classifyNB(testData,proList0,proList1,probality):
   p0=p1=0.0;
   for i in range(len(proList1)):
       p0+=(testData[i]*mt.log(proList0[i]));
       p1+=(testData[i]*mt.log(proList1[i]));

   p1+=mt.log(probality);
   p0+=mt.log(1-probality);
   if p1>p0:return 1;
   else: return 0;

if __name__=="__main__":
    #data='This is a not beautiful world.';
    #listTokens=textParge(data);
    #print(listTokens);
    list1=[10,100,1000];
    data=mt.log(mt.e);
    print(data);