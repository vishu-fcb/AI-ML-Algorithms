# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:33:43 2024

@author: Vishal Mishra
"""

import tensorflow as tf
from tensorflow.keras import datasets , layers, models 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 


#Loading the cifar 10 data set 

(X_train,Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)


#Visualizing sample data set

Y_train = Y_train.reshape(-1,)
print(Y_train[:5])
classes = ["airplane", "automobile","bird", "cat","deer","dog","frog","horse","ship","truck" ]

def print_sample(X,Y,index):    
    plt.imshow(X[index])
    plt.xlabel(classes[Y[index]])
    
# print_sample(X_train , Y_train , 5)
    
#Data Normalization
X_train = X_train/255

X_test = X_test/255

# #CNN Model 

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (2,2) , activation= 'relu', input_shape = [32,32,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (2,2), activation= 'relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides=1))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

cnn.add(tf.keras.layers.Dropout(0.2))

cnn.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))

cnn.add(tf.keras.layers.Dropout(0.2))

cnn.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))

learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-6)

cnn.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   
cnn.fit(X_train, Y_train, batch_size=32,
                  epochs=30,
                  validation_data=(X_test, Y_test),
                  callbacks=[learning_rate_scheduler])  

cnn.evaluate(X_test,Y_test)