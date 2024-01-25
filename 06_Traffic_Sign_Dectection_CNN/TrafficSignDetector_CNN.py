# -*- coding: utf-8 -*-
"""
Created on 07.01.2024 15:01:26 2024

"""

### Importing the libraries

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os 
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras

print(tf.__version__)

### Categories definition

val_categories = len(os.listdir('dataset/Train'))
print(val_categories)

### Category labels definition 

class_labels = { 0:'Speed limit 20km/h',
            1:'Speed limit 30km/h', 
            2:'Speed limit 50km/h', 
            3:'Speed limit 60km/h', 
            4:'Speed limit 70km/h', 
            5:'Speed limit 80km/h', 
            6:'End of speed limit 80km/h', 
            7:'Speed limit 100km/h', 
            8:'Speed limit 120km/h', 
            9:'No passing', 
            10:'No passing vehicle over 3.5tons', 
            11:'Priority coming soon', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh more than 3.5tons not allowed', 
            17:'No entry', 
            18:'General Warning', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve ahead', 
            22:'BUmpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road construction', 
            26:'Traffic signal warning', 
            27:'Pedestrians warning', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Ice/snow warning',
            31:'Wild animals crossing', 
            32:'End of speed/passing restrcition', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Mandatroy straight', 
            36:'Mandatroy straight or right', 
            37:'Mandatroy straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing zone', 
            42:'End of no passing zone for vehicle > 3.5 tons' }

### Data preprocessing stage

training_datagenerator = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

image_data = []
image_labels = []

for i in range(val_categories):
    path = 'dataset' + '/Train/' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((64, 64))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)

# Changing the list to numpy array
            
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(image_data.shape, image_labels.shape)

### Splitting the data set into training and test step

X_train, X_test, y_train, y_test = train_test_split(image_data, image_labels, test_size=0.2, random_state=0, shuffle=True)

X_train = X_train/255 
X_test = X_test/255

y_train = tf.keras.utils.to_categorical(y_train, val_categories)
y_test = tf.keras.utils.to_categorical(y_test, val_categories)

##### Training the Model

### Building the CNN

cnn = tf.keras.models.Sequential()

### Add the convolutional layer

cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))

### Add the maxpool layer

cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

### Add second convolutional and pool layer 

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))

### Flatting step

cnn.add (tf.keras.layers.Flatten())

### Adding the first neurons layers

cnn.add(tf.keras.layers.Dense(units = 512, activation= 'relu'))

### Adding the output layer

cnn.add(tf.keras.layers.Dense(units=43, activation = 'softmax'))

### CNN Compilation step

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ### Training the CNN on the training set and evlaute on test set

cnn.fit(training_datagenerator.flow(X_train, y_train, batch_size=32), epochs=25, validation_data=(X_test, y_test))

### Validate the model on test data

from sklearn.metrics import accuracy_score

test = pd.read_csv('dataset' + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data = []

for img in imgs:
    try:
        image = cv2.imread('dataset' + '/' + img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((64, 64))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)

X_test = np.array(data)
X_test = X_test / 255

# Make predictions and convert to class indices
pred = cnn.predict(X_test)
pred_class_indices = np.argmax(pred, axis=1)

# Convert labels to one-hot encoding
labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=43)

# Calculate accuracy with the test data
accuracy = accuracy_score(np.argmax(labels_onehot, axis=1), pred_class_indices) * 100
print('Test Data accuracy: ', accuracy)
