# -*- coding: utf-8 -*-
"""
Created on Mon Dec 02 11:21:03 2024

@author: Vishal Mishra
"""

###Library Import
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

###Import Data file
data = pd.read_csv('tweet_data.csv')

### Plot the data distribution of sentiments
sentiment_count = data["sentiment"].value_counts()
plt.pie(sentiment_count, labels=sentiment_count.index,autopct='%1.1f%%',shadow=True,startangle=140)
###plt.show()

### Creating dataset X and Y
x = data.iloc[:,1]
###print(x)

### Workd cloud representation for positive words
pos_tweets = data[data["sentiment"]=="positive"]
txt = " ".join(tweet.lower() for tweet in pos_tweets["tweet_text"])
wordcloud = WordCloud().generate(txt)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
###plt.show()

### Workd cloud representation for negative words
neg_tweets = data[data["sentiment"]=="negative"]
txt = " ".join(tweet.lower() for tweet in neg_tweets["tweet_text"])
wordcloud = WordCloud().generate(txt)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
###plt.show()

### Text Normalization simple functions (Remove unuseful info, Tokenization, Stemming, Lemmatization )
tweet = "RT @AIOutsider I love this! 👍 https://AIOutsider.com #NLP #Fun"

###Delete term RT (Retweet)
def replace_retweet(tweet, default_replace=""):
    tweet = re.sub('RT\s+',default_replace, tweet)
    return tweet 

print("Processed tweet: {}".format(replace_retweet(tweet)))

### Replace usertag with term "twitteruser"
def replace_usertag(tweet, default_replace="twitteruser"):
    tweet = re.sub('\B@\w+',default_replace, tweet)
    return tweet 

print("Processed tweet: {}".format(replace_usertag(tweet)))


###Replace emoji symbols with corresponding text
import emoji
def demojize(tweet):
    tweet = emoji.demojize(tweet)
    return tweet

print("Processed tweet: {}".format(demojize(tweet)))

###Remove URLS
def replace_URL(tweet, default_replace=""):
    tweet = re.sub('(http|https):\/\/\S+',default_replace, tweet)
    return tweet 

print("Processed tweet: {}".format(replace_URL(tweet)))
 
###Remove Hashtags
def replace_hash(tweet, default_replace=""):
    tweet = re.sub('#+',default_replace, tweet)
    return tweet 

print("Processed tweet: {}".format(replace_hash(tweet)))


### Word feature processing

tweet = "LOOOOOOOOK at this ... I'd like it so much!"

### Covert all text to lower case 
def to_lowercase(tweet):
    tweet = tweet.lower()
    return tweet

print("Processed tweet: {}".format(to_lowercase(tweet)))

### Handle word repetition for example LOOOOOOOOOK -> LOOK 
def word_repetition(tweet):
    tweet = re.sub(r'(.)\1+',r'\1\1',tweet)
    return tweet

print("Processed tweet: {}".format(word_repetition(tweet)))

### Handle punctuation repetition for example !!!! -> !
def punct_repetition(tweet, default_replace=""):
  tweet = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default_replace, tweet)
  return tweet

print("Processed tweet: {}".format(punct_repetition(tweet)))

### Handle contractions
import contractions

def fix_contractions(tweet):
    tweet = contractions.fix(tweet)
    return tweet

print("Processed tweet: {}".format(fix_contractions(tweet)))


### Tokenization (splitting long sentences in words , removing punctualtion, stop words or numbers)
import string
import nltk
from nltk.tokenize import word_tokenize

print(string.punctuation)

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

stop_words.discard('not')  ### Keeping the stopword 'NOT' as it is important for sentiment analysis
print(stop_words)

###Custom function to tokenize a give sentence ! Remove punctuation, alpha numerals and remove stop words

def custom_tokenize(tweet,
                    keep_punct = False,
                    keep_alnum = False,
                    keep_stop = False):
  
  token_list = word_tokenize(tweet)

  if not keep_punct:
    token_list = [token for token in token_list
                  if token not in string.punctuation]

  if not keep_alnum:
    token_list = [token for token in token_list if token.isalpha()]
  
  if not keep_stop:
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')
    token_list = [token for token in token_list if not token in stop_words]

  return token_list


tweet = "these are 5 different words!"

### Different output variations based on different parameter values for punctuation,alnum and stop words

print("Tweet tokens: {}".format(custom_tokenize(tweet, 
                                                keep_punct=True, 
                                                keep_alnum=True, 
                                                keep_stop=True)))
print("Tweet tokens: {}".format(custom_tokenize(tweet, keep_stop=True)))
print("Tweet tokens: {}".format(custom_tokenize(tweet, keep_alnum=True)))

### Stemming : Shorten words (Trying all the common stemming alogirhtms )
### List of words to be stemmed
tokens = ["manager", "management", "managing"]

### Inititliaze the stemmer classes 
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snoball_stemmer = SnowballStemmer('english')

### Function to pass the list of words to be stemmed along with the stemming algo to be applied 
def stem_tokens(tokens, stemmer):
  token_list = []
  for token in tokens:
    token_list.append(stemmer.stem(token))
  return token_list

print("Porter stems: {}".format(stem_tokens(tokens, porter_stemmer)))
print("Lancaster stems: {}".format(stem_tokens(tokens, lancaster_stemmer)))
print("Snowball stems: {}".format(stem_tokens(tokens, snoball_stemmer)))


### Lemmatization : Similar to stemming but makes use of word context (Dictionary based words are returned)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')

tokens = ["international", "companies", "had", "interns"]

### Part of speech tagging (needed for Lemmatization)
word_type = {"international": wordnet.ADJ, 
             "companies": wordnet.NOUN, 
             "had": wordnet.VERB, 
             "interns": wordnet.NOUN
             }

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens, word_type, lemmatizer):
  token_list = []
  for token in tokens:
    token_list.append(lemmatizer.lemmatize(token, word_type[token]))
  return token_list


print("Tweet lemma: {}".format(
    lemmatize_tokens(tokens, word_type, lemmatizer)))


### Implementing all the above custom functions on a complex example ! 

complex_tweet = r"""RT @AIOutsider : he looooook, 
THis is a big and complex TWeet!!! 👍 ... 
We'd be glad if you couldn't normalize it! 
Check https://t.co/7777 and LET ME KNOW!!! #NLP #Fun"""

def process_tweet(tweet):
   
   tweet =replace_retweet(tweet) ### Remove RT from the text 
   tweet =replace_usertag(tweet)  ### Replace username with term twitteruser
   tweet =demojize(tweet) ### Replace thumbs up emoji with text
   tweet =replace_URL(tweet) ### Remove URL
   tweet =replace_hash(tweet) ### Remove HASH
   ###print("Post Twitter processing tweet: {}".format(tweet))

   ### Implementing word features
   tweet =to_lowercase(tweet) ### Convert tweet to lower case
   tweet =word_repetition(tweet) ### Remove repeated characters like loooook
   tweet =fix_contractions(tweet) ### Fix contractions 
   tweet = punct_repetition(tweet) ### Replace punctuation repetition
   tweet = word_repetition(tweet) ### Replace word repetition
   ###print("Post Word processing tweet: {}".format(tweet))
 
   ### Apply Stemmer technique to fetch the stem words ! 
   tokens = custom_tokenize(tweet, keep_alnum=False, keep_stop=False)
   stemmer = SnowballStemmer("english")
   stem = stem_tokens(tokens, stemmer)

   return stem

###print(process_tweet(complex_tweet))

### Text Representation techniques , converting dependent variable Y to integer 1 and 0 and raw tweets to standard proceesed tweet. 
data["tokens"] = data["tweet_text"].apply(process_tweet) 
data["tweet_sentiment"] = data["sentiment"].apply(lambda i: 1 if i == "positive" else 0) 

###print(data.head(10))

###Prepare dataset
X = data["tokens"].tolist()
y = data["tweet_sentiment"].tolist()

#### Create Bag of words for the input data ( Bag of words : Converting text into a vector)

# Combine tokens back into strings for CountVectorizer
X_strings = [" ".join(tokens) for tokens in X]
###print(X_strings)

### Countvectorizer class to create the vector representation for each array
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2) 
X_vectorized = vectorizer.fit_transform(X_strings)
print(X_vectorized)

# Split the data set in training and test data ( 80% training data and 20% is test data)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

### Defition of a grid of hyperparameters for the LogisticRegression model
### C : Inverse of regularization strength , smaller value of C means more penatly annd higher C value means lower penalty
### solver : lbfgd is L2 regularization , liblinear : L1 + L2
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}

### This grid tells the GridSearchCV to try each combination of C and solver.
### GridSearchCV:
### Performs an exhaustive search over the hyperparameter combinations specified in param_grid.
### Automatically tests all combinations of C and solver on the model
#### LogisticRegression(max_iter=1500): Logistic Regression is the base model to be optimized. max_iter=1500 ensures enough iterations for convergence.
### param_grid: Defines the grid of parameters to test.
### cv=5: Uses 5-fold cross-validation. The dataset is split into 5 parts, and the model is trained and validated on different combinations of these splits.
### scoring='accuracy': Measures model performance using accuracy for each parameter combination.
grid_search = GridSearchCV(LogisticRegression(max_iter=1500), param_grid, cv=5, scoring='accuracy')
model = grid_search.fit(X_train, y_train)
### Identifies the parameter combination that achieved the highest cross-validation accuracy.
print("Best parameters:", grid_search.best_params_)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

### Improving the model by applying RandomforstClassifier with a combination of Logistic regression ( Model efficiency ~ 90%)
### 100 decision trees 
rf = RandomForestClassifier(n_estimators=100, random_state=42)

### Combines multiple "base" models (lr and rf in this case) into a single ensemble model.
### Uses the predictions of the base models as inputs to a "meta-model" (final estimator) to make the final prediction. 
stacking_model = StackingClassifier(
    estimators=[('lr', model), ('rf', rf)],
    final_estimator=LogisticRegression()
)

### Training the model based on the Stacking model
stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_test)
print(f"Stacking Ensemble Accuracy: {accuracy_score(y_test, stacking_pred)}")

### Evaluating the model efficiency
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
