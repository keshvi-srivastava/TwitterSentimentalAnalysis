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
from sklearn.linear_model import RidgeClassifier
import pickle

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

def lemmatization(tweet):
    
    tweet_list = tweet.split()
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
        
    eachTweet = " ".join(normalized_tweet) 
    
    return eachTweet

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
    
    negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

    # Remove negative patterns
    text = neg_pattern.sub(lambda x: negations_dic[x.group()], text)
    
    # Remove Hashtags & Numbers
    text = re.sub("[^a-zA-Z]", " ", text)
    
    # remove extra white spaces
    text = re.sub(r'\s+', r' ', text)
    
    text = removeStopWords(text)
    
    text = lemmatization(text)
    
    return text

def main():

    #Read and format the excel data
    data_obama = pd.read_excel('training-Obama-Romney-tweets.xlsx', 'Obama', header=1)
    data_obama.rename(columns = {'Unnamed: 1':'date', 'Unnamed: 2':'time', 
                            '1: positive, -1: negative, 0: neutral, 2: mixed':'Anootated tweet'}, inplace = True)
    data_obama = data_obama.drop(['Unnamed: 0','date', 'time', 'Your class', 'Unnamed: 6'], axis=1)
    data_obama['Class'] = data_obama['Class'].apply(str)
    data_obama = data_obama[(data_obama['Class'] == '1') | (data_obama['Class'] == '0') | (data_obama['Class'] == '-1')]
    data_obama['Class'] = data_obama['Class'].apply(int)
    data_obama = data_obama.dropna()

    #Pre-process the data
    data_obama['Anootated tweet'] = data_obama.apply(preprocess_tweet, axis=1)

    #Split the data into training, validation and test dataset
    x = data_obama['Anootated tweet']
    y = data_obama['Class']
    SEED = 2000

    # Split data
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.1, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    print("x_train: ", len(x_train))
    print("x_validation: ", len(x_validation))
    print("x_test: ", len(x_test))

    #Set TF-IDF params
    n_features=100000
    ngram_range=(1,3)
    tvec = TfidfVectorizer()
    tvec.set_params(stop_words=None, max_features=n_features, ngram_range=ngram_range)

    target_class = ['-1', '0', '1'] 

    #Train Classifier
    clf = RidgeClassifier()
    checker_pipeline = Pipeline([
            ('vectorizer', tvec),
            ('classifier', clf)
        ])
    sentiment_fit_obama = checker_pipeline.fit(x_train, y_train)

    #Test the validation set
    y_pred_val = sentiment_fit_obama.predict(x_validation)
    print("Validation data report\n")
    print(accuracy_score(y_validation, y_pred_val))
    print(classification_report(y_validation, y_pred_val, target_names=target_class))

    #Test the test set
    y_pred_test = sentiment_fit_obama.predict(x_test)
    print("Test data report\n")
    print(accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test, target_names=target_class))

    with open('clf_obama.pickle', 'wb') as f:
        pickle.dump(sentiment_fit_obama, f)

if __name__ == "__main__":
   main()