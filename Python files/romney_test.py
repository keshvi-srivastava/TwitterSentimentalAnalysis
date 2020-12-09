import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from html import unescape
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer as porterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import pickle

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
    negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    text = neg_pattern.sub(lambda x: negations_dic[x.group()], text)
    
    # Remove Hashtags & Numbers
    text = re.sub("[^a-zA-Z]", " ", text)
    
    # remove extra white spaces
    text = re.sub(r'\s+', r' ', text)
    
    # text = removeStopWords(text)
    
    # text = lemmatization(text)
    
    return text
    
def main():
    with open('clf_romney.pickle', 'rb') as f:
        clf = pickle.load(f)

    data_romney_test = pd.read_excel('final-testData-no-label-Romney-tweets.xlsx', header=None,
                          names=['tweet_id', 'Anootated tweet'])

    target_class = ['-1', '0', '1']

    data_romney_test['Anootated tweet'] = data_romney_test.apply(preprocess_tweet, axis=1)

    data_romney_test['Predicted Class'] = clf.predict(data_romney_test['Anootated tweet'])

    f = open('romney.txt', "w")
    f.write("{} {}\n".format(57,68))
    for idx, row in data_romney_test.iterrows():
        f.write("{};;{}\n".format(row['tweet_id'],row['Predicted Class']))

    f.close()
    
if __name__ == "__main__":
   main()