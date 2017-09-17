import numpy as np
import pandas as pd
data_train=pd.read_csv('G:\digit\\train.csv')
data_test=pd.read_csv('G:\digit\\test.csv')
data_train=data_train.values
data_test=data_test.values
data_test=np.multiply(data_test,1.0/225.0)
temp=data_train[:,1:]
temp=np.multiply(temp,1.0/225.0)
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
#getncomponent(data)
pca = PCA(n_components=40,whiten=True)
x=pca.fit_transform(data)
train_x = x[0:42000,:]
train_y=data_train[:,0]
test_x = x[42000:,:]

import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(train_x[:,0],train_x[:,1],s=20,c=train_y,cmap = "nipy_spectral", edgecolor = "None")
plt.colorbar()
plt.clim(0,9)
plt.xlabel("PC1")
plt.ylabel("PC2")




from sklearn.svm import SVC
clf=SVC(C=6,kernel='poly',degree=9,probability=True)
clf.fit(train_x,train_y)
result_proba=clf.predict_proba(test_x)

result_max=np.zeros((test_x.shape[0],2))
for i in range(test_x.shape[0]):
    result_max[i,0]=max(result_proba[i,:])
    result_max[i,1]=np.argmax(result_proba[i,:])

for i in range(10):
    print(sum(result_max[(result_max[:,0]<0.99),1]==i))


#save proba result
import csv
with open("G:\\digit\\SVM_poly_1_0_2_4_7.csv","w") as f:
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