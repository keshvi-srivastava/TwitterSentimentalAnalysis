# Project Report
## Course - CS 583 (Data Mining and Text Mining)
## Term - Fall 2020

## Team - 
- Keshvi Srivastava           652825616
- Sneha Mysore Suresh   677216079

### 1. Introduction

Social media platforms, such as Facebook and Twitter, have not only changed the way we interact with one another but also the way we share news and comment on world events.Twitter has become an increasingly relevant tool in domestic and international politics. The platform has become a way to promote policies and interact with citizens and other officials.Sentiment analysis of Twitter data is becoming a popular research area. 

In this project we built models for classifying tweets about Obama and Romney during the presidential debate from 2012 into positive, negative and neutral sentiments. We have studied various approaches to work with tweet analysis which are explained below. Our study excludes class '2' tweets which come under the category of sarcasm detection.

### 2. Dataset and pre-processing

Obama -  Total : 7196 tweets, Removing Class 2  : 5624 tweets

Romney - Total : 7200 tweets, Removing Class 2  : 5648 tweets

Tweet Pre-processing -

Cleaning the data to remove irrelevant words and character is a big part of text classification. We studied our data to identify what problems exist within it. We applied standard tweet pre-processor cleaning libraries but those were not enough. To compensate for the remaining dirty data, we have done additional cleaning. All the cleaning processes are given below:

1. Tweet pre-processing class (tweet-preprocessor) removes: 
      a. URLs
      b. Hashtags
      c. Mentions
      d. Reserved words (@RT, FAV)
      e. Emojis 
      f. Smileys
      g. JSON and .txt file support
2. Remove explicit character specified smileys
3. Remove html tags
4. Remove punctuations
5. Remove white space
6. Split negative words such as 'don't' to 'do not'. This turns the data into a uniform format which does not have a mixture of shortened words and their split words. It also highlights the negative words such as 'not'.
7. Remove unicodes
8. Remove stop words -> Apart from the standard nltk library stop words, we also added 'RT' as a stop word to the list (it stands for retweet and was not removed by the tweet pre-processor because it did not follow '@' in the tweets)
9. Lemmatization

### 3. Data Visualisation

For data visualisation we studied the Obama and Romney word clouds and Top 50 words of each class. We observed that for both Obama and Romney the two respective names occur the most in their respective word clouds, whether positive, negative or neutral. Also, a lot of words tend to overlap in the different classes of tweets which made it very difficult to use 'bag of words' for our classification models due to very small number of unique words.

**I. Word clouds**


<img src="images/obama_neg.png" width="425" /> Obama Negative Word Cloud
<img src="images/obama_neut.png" width="425" /> Obama Neutral Word Cloud
<img src="images/obama_pos.png" width="425" /> Obama Positive Word Cloud
<img src="images/romney_neg.png" width="425" /> Romney Negative Word Cloud
<img src="images/romney_neut.png" width="425" /> Romney Neutral Word Cloud
<img src="images/romney_pos.png" width="425" /> Romney Positive Word Cloud

**II. Top 50 negative, neutral and positive words**

<img src="images/obama_neg_count.png" width="425" /> Obama Negative Word Count
<img src="images/obama_neut_count.png" width="425" /> Obama Neutral Word Count
<img src="images/obama_pos_count.png" width="425" /> Obama Positive Word Count
<img src="images/romney_neg_count.png" width="425" /> Romney Negative Word Count
<img src="images/romney_neut_count.png" width="425" /> Romney Neutral Word Count
<img src="images/romney_pos_count.png" width="425" /> Romney Positive Word Count

### 4. Models tried

1. TextBlob : TextBlob is a python library to perform basic NLP tasks. It predicts text sentiment based on built-in knowledge.
2. Logistic Regression
3. Linear SVC
4. LinearSVC with L1-based feature selection
5. Multinomial NB
6. Bernoulli NB
7. Ridge Classifier
8. AdaBoost
9. Perceptron
10. Nearest Centroid
11. Ensemble : LogisticRegression, LinearSVC, MultinomialNB, RidgeClassifier, PassiveAggressiveClassifier
12. Doc2Vec + Linear regression : Doc2Vec uses a 3 layered shallow deep neural network to gauge the context of the document and relate similar context phrases together. Different variations of Doc2Vec used Distributed Bag Of Words(DBOW), Distributed Memory Concatenation(DMC), Distributed Memory Mean(DMM)

Tfid Vectorizer : TF-IDF Vectorizer converts documents to a matrix of TF-IDF (term frequency - inverse document frequency) features. We fed the following parameters to the it - (n_features=100000,ngram_range=(1,3),stop_words=None). This matrix is fed to the above models for sentiment prediction.

**OBAMA**

Training set:   5342,	Test set:  282

|                 Models                     |    Accuracy   |
|--------------------------------------------|:-------------:|
| TextBlob                                   |      43%      |
| Logistic Regression                        |     63.12%    |
| Linear SVC                                 |     63.48%    |
| LinearSVC with L1-based feature selection  |     59.93%    |
| Multinomial NB                             |     64.18%    |
| **Ridge Classifier**                       |   **64.54%**  |
| AdaBoost                                   |     44.33%    |
| Perceptron                                 |     60.64%    |
| Nearest Centroid                           |     59.93%    |
| Ensemble                                   |     60.78%    |
| Doc2Vec                                    |     57.80%    |

**ROMNEY**

Training set:   5083,	Test set:  565

|                 Models                     |    Accuracy   |
|--------------------------------------------|:-------------:|
| TextBlob                                   |      43%      |
| Logistic Regression                        |     58.30%    |
| Linear SVC                                 |     62.54%    |
| **LinearSVC with L1-based feature selection**  |     **63.25%**    |
| Multinomial NB                             |     47.35%    |
| Ridge Classifier                           |     60.78%    |
| AdaBoost                                   |     50.88%    |
| Perceptron                                 |     62.19%    |
| Nearest Centroid                           |     60.07%    |
| Ensemble                                   |     60.78%    |
| Doc2Vec                                    |     53.90%    |

### 5. Results

We studied the above models and applied different combinations of data pre-processing. Interestingly the accuracy of the Romney model improved by not removing the stop words or doing lemmatization on the tweets. These were our final results:

|                     |               Obama                          |                      Romney                  |
|---------------------|:--------------------------------------------:|:--------------------------------------------:|
|  Pre-processing     |    stop words and lemmatization included     |  stop words and lemmatization not included   |
|      Model          |            Ridge Classifier                  | Linear SVC with feature selection            |
|    **Precision**        |
|     Class 1         |                 0.64                         |                        0.61                  |
|     Class -1        |                 0.69                         |                        0.65                  |
|    **Recall**           |                                              |                                              |
|     Class 1         |                 0.75                         |                        0.59                  |
|     Class -1        |                 0.75                         |                        0.81                  |
|    **F1 Score**         |                                              |                                              |
|     Class 1         |                 0.58                         |                        0.60                  |
|     Class -1        |                 0.61                         |                        0.72                  |
|  **Weighted Accuracy**  |                 0.67                         |                        0.63                  |

### 6. Conclusion

In this project, we experimented with various models for the tweet classification. We noticed that since twitter data is inconsistent - some tweets may include emojis, special characters and stop words related to tweets (@,#,RT) - data preprocessing was a crucial step before applying any of the models. We tried a range of other models including simple classifiers, emsemble techniques and neural network models. Since the dataset was relatively small, the deep neural network models we tried (Doc2Vec with it's vaiations) did not yeild great results.Finally, *Ridge Classifier* gave us the best results for Obama data and *Linear SVC with feature selection* gave us the best results for Romney data.
