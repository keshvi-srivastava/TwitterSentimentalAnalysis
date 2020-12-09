#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
# nltk.download()


# In[2]:


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer as porterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.svm import SVC
from datetime import datetime
import matplotlib.pyplot as plt
import preprocessor as p
import string
from bs4 import BeautifulSoup
from html import unescape
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import wordnet as guru
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from sklearn import utils
from nltk.stem import WordNetLemmatizer


# ## Read Data

# In[3]:


data_obama = pd.read_excel('training-Obama-Romney-tweets.xlsx', 'Obama', header=1)
data_obama.rename(columns = {'Unnamed: 1':'date', 'Unnamed: 2':'time', 
                            '1: positive, -1: negative, 0: neutral, 2: mixed':'Anootated tweet'}, inplace = True)


# In[4]:


data_obama = data_obama.drop(['Unnamed: 0','date', 'time', 'Your class', 'Unnamed: 6'], axis=1)
data_obama.head()


# In[5]:


data_obama.count()


# ## Filter out Class 2

# In[6]:


data_obama['Class'] = data_obama['Class'].apply(str)

data_obama = data_obama[(data_obama['Class'] == '1') | (data_obama['Class'] == '0') | (data_obama['Class'] == '-1')]


# In[7]:


data_obama.count()
data_obama['Class'] = data_obama['Class'].apply(int)


# In[8]:

data_obama = data_obama.dropna()



# ## Count of observation

# In[9]:


negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')


# # OBAMA

# In[10]:

def removeStopWords(tweet):
    
    filtered_tweet = [];

    #stemming
    porter = porterStemmer()
    stemmedTweet = [porter.stem(word) for word in tweet.split(" ")]
    stemmedTweet = " ".join(stemmedTweet)
    tweet = str(stemmedTweet);
    
    tweet = tweet.replace("'", "");
    
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(['RT'])
    
    word_tokens = word_tokenize(tweet)
    
    for w in word_tokens:
        if w not in stop_words:
            filtered_tweet.append(w)
    
    eachTweet = " ".join(filtered_tweet)  
    
    return eachTweet

#Built in tweet cleaner


# In[11]:

def lemmatization(tweet):
    
    tweet_list = tweet.split()
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
        
    eachTweet = " ".join(normalized_tweet) 
    
    return eachTweet


# In[12]:


def preprocess_tweet(row):
    
    text = row['Anootated tweet']
    
    # HTML Decoding
    soup = BeautifulSoup(unescape(text), 'lxml')
    text = soup.text
    
    # Remove emojis
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' EMO_POS ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', text)
    
    # Removing @
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    
    # Remove URL links
    text = re.sub('https?://[A-Za-z0-9./]+','',text)
    text = re.sub(r'www.[^ ]+', '', text)
    
    # Lower Case
    text = text.lower()
    
    #Remove words with repetition greater than 2
    word = re.sub(r'(.)\1+', r'\1\1', text)
    
    # Remove negative patterns
    text = neg_pattern.sub(lambda x: negations_dic[x.group()], text)
    
    # Remove Hashtags & Numbers
    text = re.sub("[^a-zA-Z]", " ", text)
    
    # remove extra white spaces
    text = re.sub(r'\s+', r' ', text)
    
    text = removeStopWords(text)
    
    text = lemmatization(text)
    
    return text


# In[13]:


data_obama['Anootated tweet'] = data_obama.apply(preprocess_tweet, axis=1)

data_obama


# In[14]:


data_obama = data_obama[data_obama['Anootated tweet'] != '']


# In[15]:




# In[16]:




# In[16]:


annotatedTweet = data_obama['Anootated tweet']
actualClass = data_obama['Class']
lengthObama = len(data_obama);

annotatedTweetList = annotatedTweet.tolist();
actualClassList = actualClass.tolist();
print(annotatedTweetList)




# In[18]:

from sklearn.model_selection import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(annotatedTweet,actualClass, test_size=.1, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

# In[19]:

def labelize_tweets_ug(tweets,label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(LabeledSentence(t.split(), [prefix + '_%s' % i]))
    return result
  


# %%

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')


# %%
cores = multiprocessing.cpu_count()

model_ug_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)])

# %%time


for epoch in range(30):
    model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dbow.alpha -= 0.002
    model_ug_dbow.min_alpha = model_ug_dbow.alpha
# %%

def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs


# %%

train_vecs_dbow = get_vectors(model_ug_dbow, x_train, 100)
validation_vecs_dbow = get_vectors(model_ug_dbow, x_validation, 100)
test_vecs_dow = get_vectors(model_ug_dbow, x_test, 100)
print(test_vecs_dow)

# %%

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_vecs_dbow, y_train)

# %%
clf.score(validation_vecs_dbow, y_validation)

# %%
clf.score(test_vecs_dow, y_test)
# %%
#LinearSVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))])
clf.fit(train_vecs_dbow, y_train)

# %%
clf.score(validation_vecs_dbow, y_validation)
# %%
clf.score(test_vecs_dow, y_test)
# %%
model_ug_dbow.save('d2v_model_ug_dbow.doc2vec')
model_ug_dbow = Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
# %%
# model_ug_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
# %%
#Distributed Memory (concatenated)
cores = multiprocessing.cpu_count()
model_ug_dmc = Doc2Vec(dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_dmc.build_vocab([x for x in tqdm(all_x_w2v)])
# %%
%%time
for epoch in range(30):
    model_ug_dmc.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmc.alpha -= 0.002
    model_ug_dmc.min_alpha = model_ug_dmc.alpha
# %%

train_vecs_dmc = get_vectors(model_ug_dmc, x_train, 100)
validation_vecs_dmc = get_vectors(model_ug_dmc, x_validation, 100)
test_vecs_dmc = get_vectors(model_ug_dbow, x_test, 100)
# %%
clf = LogisticRegression()
clf.fit(train_vecs_dmc, y_train)

# %%
clf.score(validation_vecs_dmc, y_validation)
# %%
clf.score(test_vecs_dmc, y_test)
# %%
#Distributed Memory (mean)
cores = multiprocessing.cpu_count()
model_ug_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_dmm.build_vocab([x for x in tqdm(all_x_w2v)])
    
# %%
%%time
for epoch in range(30):
    model_ug_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_dmm.alpha -= 0.002
    model_ug_dmm.min_alpha = model_ug_dmm.alpha
# %%

train_vecs_dmm = get_vectors(model_ug_dmm, x_train, 100)
validation_vecs_dmm = get_vectors(model_ug_dmm, x_validation, 100)
test_vecs_dmm = get_vectors(model_ug_dbow, x_test, 100)
# %%
clf = LogisticRegression()
clf.fit(train_vecs_dmm, y_train)
# %%
clf.score(validation_vecs_dmm, y_validation)
# %%
clf.score(test_vecs_dmm, y_test)
# %%
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))])
clf.fit(train_vecs_dmm, y_train)

# %%
clf.score(validation_vecs_dmm, y_validation)
# %%
clf.score(test_vecs_dmm, y_test)
# %%
