# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from scipy import sparse
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#threshold=0.5,0.6,0.7,0.75,0.8,0.9
def nBayesClassifier(traindata,trainlabel,testdata,testlabel,threshold):
    clf = MultinomialNB()
    clf.fit(traindata,trainlabel)
    ypred= clf.predict_proba(testdata)
    ypred=np.where(ypred[:,1]>threshold,1,-1)
    count=0
    for i in range(len(testlabel)):
        if testlabel[i]==ypred[i]:
            count=count+1
    accuracy=float(count)/len(testlabel)
    #accuracy = metrics.precision_score(testlabel, np.array(ypred))
    #recall = metrics.recall_score(testlabel,  np.array(ypred))
    return ypred,accuracy

#lambda=1e-4,0.01,0.1,1,0.5,1,10,100,1000,5000,10000
def lsClassifier(traindata,trainlabel,testdata,testlabel,Lambda):
    reg = linear_model.Ridge(alpha=Lambda)
    reg.fit(traindata,trainlabel)
    ypred=reg.predict(testdata)
    ypred = np.where(ypred > 0, 1, -1)
    count = 0
    for i in range(len(testlabel)):
        if testlabel[i] == ypred[i]:
            count = count + 1
    accuracy = float(count) / len(testlabel)
    return ypred, accuracy

#singma=0.01d,0.1d,d,10d,100d
#c=1,10,100,1000
def softsvm(traindata,trainlabel,testdata,testlabel,sigma,c):
    if sigma==0:
        clf = svm.SVC(kernel='linear',C=c)
    else:
        clf=svm.SVC(kernel='rbf',gamma=1.0/(sigma*sigma),C=c)
    clf.fit(traindata, trainlabel)
    ypred=clf.predict(testdata)
    count = 0
    for i in range(len(testlabel)):
        if testlabel[i] == ypred[i]:
            count = count + 1
    accuracy = float(count) / len(testlabel)
    return ypred,accuracy

def getFeature(comment): # read a comment
    row=[]
    col=[]
    data=[]
    comment=comment[1:]
    for i in range(len(comment)):
        index_value=comment[i].split(":")
        index=int(index_value[0])
        value = int(index_value[1])
        col.append(index)
        data.append(value)
        row.append(0)
    vcom = sparse.coo_matrix((data, (row, col)), shape=(1, 43109))
    v = np.array(vcom.toarray())
    v_normalized = preprocessing.normalize(v, norm='l1')
    return v_normalized[0]

trainlabel=[]
count=0
row = []
col = []
data = []
idf= np.loadtxt(open("idf.txt","r"),delimiter=",")
with open("data") as file:
    for line in file.readlines():
        line=line.strip().split(' ')
        label = int(line[0])
        if label >= 7:
            label = 1
        elif label <= 4:
            label = -1
        else:
            label = 0
        trainlabel.append(label)
        #v=getFeature(line)
        #v=v.tolist()
        #traindata.append(v)
        comment = line[1:]
        for i in range(len(comment)):
            index_value = comment[i].split(":")
            index = int(index_value[0])
            value = int(index_value[1])
            col.append(index)
            data.append(value*idf[index])
            row.append(count)
        count=count+1

traindata= sparse.coo_matrix((data, (row, col)), shape=(50000, 89527))
traindata= preprocessing.normalize(traindata, norm='l1')
trainlabel=np.array(trainlabel)
print("finish")
X_train, X_test, y_train, y_test = train_test_split(traindata, trainlabel, test_size=0.2)
#a,b=nBayesClassifier(X_train,y_train,X_test,y_test,0.5)
#a,b=lsClassifier(X_train,y_train,X_test,y_test,0.1)
a,b=softsvm(X_train,y_train,X_test,y_test,3.3,1)
#print(a)
print(b)
'''
d=0.033
sigma=[0.01*d,0.1*d,d,10*d,100*d] 
c=[1,10,100,1000]
svm_accuracy=np.zeros((len(sigma),len(c)))
for i in range(len(sigma)):
    for j in range(len(c)):
        ypred, svm_accuracy[i][j]= softsvm(X_train, y_train, X_test, y_test, sigma[i],c[j])
        print(svm_accuracy[i, j])
print svm_accuracy


# 定义分类器
clf= MultinomialNB()
# 进行交叉验证数据评估, 数据分为5部分, 每次用一部分作为测试集
nb_scores = cross_val_score(clf, traindata, trainlabel, cv = 5, scoring = 'accuracy')
# 输出5次交叉验证的准确率
print nb_scores

clf=svm.SVC(kernel='linear',C=0.1)
svm_scores = cross_val_score(clf, traindata, trainlabel, cv = 5, scoring = 'accuracy')
print svm_scores
'''

