"""
Created on 02.01.2024

@author: Vishal Mishra
"""

import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

### Check File encoding format

file = 'spam.csv'
import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

### Read the CSV in the right format

dataset = pd.read_csv(file,encoding='Windows-1252')
Y = dataset.iloc[:,0].values
X = dataset.iloc[:,1].values

#print(X.size)
#print(Y.size)

### NLP on dependent variable X

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
modified_text = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ', X[i])     
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    modified_text.append(review)
  
# print(modified_text)

### Create Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(modified_text).toarray()

print(len(X[0]))

### Label Encoding the dependent variable Y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y= le.fit_transform(Y)
print(Y)

### Splitting the data set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2, random_state= 1)


###### Building the ANN 
### ANN Initialization
ann = tf.keras.models.Sequential()

### Adding the input layer and first hidden layer 
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

###Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation= 'relu'))

###Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

### Compile the ANN
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

### Train the ANN
ann.fit(X_train, Y_train, batch_size = 100, epochs = 200)

### Predicting the test set
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))


### Confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(accuracy_score(Y_test, y_pred))



### Sample Example with a random message

new_review = input('Write your email format\n')
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = ann.predict(new_X_test)
print('spam or valid')
if (new_y_pred [0] > .5):
    print('Genuine E-mail')
else:
    print('Spam E-mail')


