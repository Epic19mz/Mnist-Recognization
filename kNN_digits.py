# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 22:15:35 2017

@author: Zhen MIAO
"""

import numpy as np
import pandas as pd
data_train=pd.read_csv('G:\\digit\\train.csv')
data_test=pd.read_csv('G:\digit\\test.csv')
data_train=data_train.values
data_test=data_test.values
data_test=np.multiply(data_test,1.0/225.0)
temp=data_train[:,1:]
temp=np.multiply(temp,1.0/225.0)
data_train[:,1:]=temp
data=np.vstack((data_train[:,1:],data_test))
from sklearn.decomposition import PCA
def getncomponent(inputdata):
    pca = PCA()  
    pca.fit(inputdata)    
    EV_List = pca.explained_variance_  
    EVR_List = []  
    for j in range(len(EV_List)):  
        EVR_List.append(EV_List[j]/EV_List[0])  
    for j in range(len(EVR_List)):  
        if(EVR_List[j]<0.03):  
            print ('Recommend %d:' %j)
            return j
#getncomponent(data)
pca = PCA(n_components=80,whiten=True)
x=pca.fit_transform(data)
train_x= x[0:42000,:]
train_y=data_train[:,0]
test_x=x[42000:,:]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,weights='distance',p=2)
knn.fit(train_x,train_y)
result_proba=knn.predict_proba(test_x)

result_max=np.zeros((test_x.shape[0],2))
for i in range(test_x.shape[0]):
    result_max[i,0]=max(result_proba[i,:])
    result_max[i,1]=np.argmax(result_proba[i,:])

for i in range(10):
    print(sum(result_max[(result_max[:,0]<0.99),1]==i))


#save proba result
import csv
with open("G:\\digit\\kNN_1_0_1_6.csv","w") as f:
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