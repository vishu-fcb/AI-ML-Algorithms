"""
Created on Thu Dec 21 12:35:09 2023

@author: VISHMIS
"""

import numpy as np
import pandas as pd

### Check File encoding format

file = 'spam.csv'
import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

### Read the CSV in the right format

dataset = pd.read_csv(file,encoding='Windows-1252')
Y = dataset.iloc[:,0].values
X = dataset.iloc[:,1].values

print(X.size)
print(Y.size)

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
  
#print(modified_text)

### Create Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(modified_text).toarray()

print(len(X[0]))


### Split Data into training and test set
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)


### Training Support vector machine on training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

### Training Naive BAyes on training set yielded very poor results 82%
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, Y_train)


### Predicting test result 
Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))


### Confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(accuracy_score(Y_test, Y_pred))

### Sample Example with a random message

new_review = input('Write your email format\n')
#new_review = 'Hello shivangi, hoep you are dping well'
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
new_y_pred = classifier.predict(new_X_test)
print(new_review)
print('spam or valid')
print(new_y_pred)
