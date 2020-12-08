# Project Report
## Course - CS 583 (Data Mining and Text Mining)
## Term - Fall 2020

## Team - 
- Keshvi Srivastava     652825616
- Sneha Mysore Suresh   .........

### 1. Introduction

### 2. Dataset and pre-processing

Obama - 
  Total -> 7196 tweets
  Removing Class 2 -> 5624 tweets
  
 Romney -
 
 Pre-processing -
 
 1. Remove stop words -> added 'RT' as a stop word to the list as it was not relevant (it stands for retweet)
 2. Lemmatization
 3. Tweet pre-processing class
    Removes:  
      a. Hyperlinks
      b. Emojis
4. Remove unicode smileys
5. remove html tags
6. remove punctuations
7. Remove white space
8. Split negative words
9. Remove unicodes

### 3. Visualisation

Word clouds
Top 50 negative, neutral and positive words

### 4. Model tried

OBAMA

1. TextBlob - TextBlob is a python library and offers a simple API to access its methods and perform basic NLP tasks. A good thing about TextBlob is that they are just like python strings. So, you can transform and play with it same like we did in python

Accuracy Score: 42.70%

          precision    recall  f1-score
 -1       0.54      0.23      0.32
  0       0.41      0.61      0.49  
  1       0.41      0.43      0.42

    accuracy   0.43      
   macro avg   0.41     
weighted avg   0.41    

Tabular data

ROMNEY

1. TextBlob - TextBlob is a python library and offers a simple API to access its methods and perform basic NLP tasks. A good thing about TextBlob is that they are just like python strings. So, you can transform and play with it same like we did in python

Accuracy Score: 42.70%

              precision    recall  f1-score 

          -1       0.69      0.33      0.45     
           0       0.40      0.50      0.45    
           1       0.30      0.49      0.37  

    accuracy             0.43
   macro avg             0.42
weighted avg             0.43

Tabular data

### 5. Results

Obama - with all pre-processing
      - model selected -> SVM
      
Romney - without stop words and lemmatization
       - model selected -> Linear SVC with feature selection


### 6. Conclusion
 
  
