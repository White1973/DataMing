from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.np_utils import to_categorical
import numpy as np

#构建神经网络,训练模型
def NueralNet(trainData,trainLabel,testData,testLabel):
    #将三位矩阵转化为2维，每一组值由灰度图像变为一维向量
    trainData=trainData.reshape(trainData.shape[0],-1);
    testData=testData.reshape(testData.shape[0],-1);
    #需要转化为维矩阵
    col=np.array(trainData).shape[1];
    model=Sequential();
    model.add(Dense(100,input_dim=col,activation="relu"));
    model.add(Dense(50,activation="relu"));
    model.add(Dense(16,activation="softmax"));

    model.compile(optimizer="sgd",loss="categorical_crossentropy",metrics=['accuracy']);
    categorical_train_labels = to_categorical(trainLabel, num_classes=None);
    categorical_test_labels=to_categorical(testLabel, num_classes=None);
    model.fit(trainData,categorical_train_labels,batch_size=20,epochs=400);

    predict=model.predict(testData);
    return predict;

   #result=model.evaluate(testData,categorical_test_labels,batch_size=20);
    '''
    predict_test=np.array(model.predict(testData));
    count=0;
    for i in range(predict_test.shape[0]):
        if np.array(predict_test[i]).argmax()==testLabel[i]:
            count+=1;

    result=float(count)/float(predict_test.shape[0]);
    '''
    #print("准确率:",result);
    #return result;

