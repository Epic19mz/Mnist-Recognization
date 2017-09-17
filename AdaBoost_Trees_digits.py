# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:29:11 2017

@author: Zhen MIAO
"""

import numpy as np
import pandas as pd
data_train=pd.read_csv('G:\\digit\\train.csv')
data_test=pd.read_csv('G:\\digit\\test.csv')
data_train=data_train.values
data_test=data_test.values
data_test[data_test<50]=0
data_test[data_test>=50]=1
temp=data_train[:,1:]
temp[temp<50]=0
temp[temp>=50]=1
data_train[:,1:]=temp
from sklearn.decomposition import PCA
def getncomponent(inputdata):
    pca = PCA()  
    pca.fit(inputdata)    
    EV_List = pca.explained_variance_  
    EVR_List = []  
    for j in range(len(EV_List)):  
        EVR_List.append(EV_List[j]/EV_List[0])  
    for j in range(len(EVR_List)):  
        if(EVR_List[j]<0.05):  
            print ('Recommend %d:' %j)
            return j
data=np.vstack((data_train[:,1:],data_test))
getncomponent(data)
pca = PCA(n_components=36,whiten=True)
x=pca.fit_transform(data)
train_x = x[0:42000,:]
train_y=data_train[:,0]
test_x = x[42000:,:]
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
tree_model = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_depth=25,
            max_features='sqrt', 
            max_leaf_nodes=None,
            min_samples_leaf=20,
            min_samples_split=400)
ABM=AdaBoostClassifier(base_estimator=tree_model, learning_rate=0.1, n_estimators=1000,random_state=1)
ABM.fit(train_x,train_y)
result_proba=ABM.predict_proba(test_x)
result_max=np.zeros((test_x.shape[0],2))
for i in range(test_x.shape[0]):
    result_max[i,0]=max(result_proba[i,:])
    result_max[i,1]=np.argmax(result_proba[i,:])

for i in range(10):
    print(sum(result_max[(result_max[:,0]<0.99),1]==i))


#save proba result
import csv
with open("G:\\digit\\adaBoost_1.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(result_proba)

result_label=np.argmax(result_proba,axis=1)
# save results
np.savetxt('G:\\digit\\submission_softmax.csv', 
           np.c_[range(1,len(test_x)+1),result_label], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')