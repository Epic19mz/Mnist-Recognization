# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 256
num_classes = 10
epochs = 25

# separate to data and label

def opencsv(): 
    data = pd.read_csv('G:\\digit\\train.csv')
    data1 = pd.read_csv('G:\\digit\\test.csv')
    x_train = data.values[0:, 1:]
    y_train = data.values[0:, 0]
    x_test = data1.values[0:, 0:]
    print('Data Load Done!')
    return x_train, y_train, x_test
x_train, y_train, x_test = opencsv() 

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
# cov
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
# pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# dropout
model.add(Dropout(0.25))
# flatten
model.add(Flatten())
# activation
model.add(Dense(128, activation='relu'))
# dropout
model.add(Dropout(0.5))
# activation
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

result_proba=model.predict(x_test)

result_max=np.zeros((x_test.shape[0],2))
for i in range(x_test.shape[0]):
    result_max[i,0]=max(result_proba[i,:])
    result_max[i,1]=np.argmax(result_proba[i,:])

for i in range(10):
    print(sum(result_max[(result_max[:,0]<0.99),1]==i))

import csv
with open('G:\\digit\\keras_1_.csv','w') as f:
    writer=csv.writer(f)
    writer.writerows(result_proba)
result_lables=np.argmax(result_proba,axis=1)

with open('G:\\digit\\submission.csv', 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(result_lables)) :
        f.write(''.join([str(i+1),',',str(result_lables[i]),'\n']))