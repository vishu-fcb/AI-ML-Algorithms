# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:21:03 2024

@author: Vishal Mishra
"""

###Library Import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split


###Import Data file

data = pd.read_csv('IMDB Dataset.csv')
data.replace(to_replace=r'<br />', value='', regex=True, inplace=True)
print("Columns in the DataFrame:", data.columns)


###Process Data in File
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stem_word = []

for i in range(0,len(data)):
    review1 = re.sub('[^a-zA-Z]', ' ' , data['review'][i])
    review1 = review1.lower()
    review1 = review1.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review1 = [ps.stem(word) for word in review1 if not word in set(all_stopwords)]
    review1 = ' '.join(review1)
    stem_word.append(review1)
#print(stem_word)    

### Encoding the Indepedent variable 
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])
y = data['sentiment'].values
#print("Encoded target variable:\n", y)


### Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(stem_word).toarray()

print(len(X[0]))

### Split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.33,random_state=0)
print(X_train)
print(Y_train)




### Start building the ANN Model 
# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

