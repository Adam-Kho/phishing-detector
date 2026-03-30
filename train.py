# Adam 
# File for training model

import pandas as pd
import matplotlib.pyplot as plt
import pickle
#import predict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

# for Naive Bayes
#from sklearn.naive_bayes import MultinomialNB

# for Logistic Regression
from sklearn.linear_model import LogisticRegression

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv('emails.csv')

def display_data(df):
    print(df.head()) # show first 5 rows
    print(df.shape)  # show (rows, columns)
    print(df['spam'].value_counts()) # 0 = legit | 1 = spam
    return

X = df['text'] # input data, text
y = df['spam'] # answers, 0 or 1
# x is like the questions and y is the answers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# text_size is .2 meaning 20% test, 80% train

# can improve this with parameters
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, token_pattern=r'[a-zA-Z]+')
X_train_tfidf = vectorizer.fit_transform(X_train) # learn vocabulary AND convert to numbers
X_test_tfidf = vectorizer.transform(X_test) # convert only, vocabulary already learned

# print(X_train_tfidf.shape) ( __ , __)

model = LogisticRegression()

# With keyword automatically closes file when the block is done. 
# Without that you need f.close()

# print("\nModel & Vectorizer saved.\n")

# model.fit trains the model, looks at TF-IDF numbers alongide the correct labels
# and learns the patterns that seperate spam vs legit
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# - pickle serializes Python objects into binary files
# meaning it converts an object in memory into a format that can be
# stored and reloaded later exactly as it was
# - retraining data every time is inefficient. Benefit is that we train 
# once, save and load instantly moving forward
# - model.pkl contains the trained Logistic Regression model with all its 
# learned weights. vectorizer.pkl contains the fitted TF-IDF vectorizer
# with its full learned vocabulary of 33,790
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Reading Classification Report
# Columns: precision (correct when flagged) | recall (caught of all actual) | f1-score (balance of both) | support (actual count in test set)
# Row 0 (legit) : how well model handles legit emails
# Row 1 (spam) : how well model handles spam emails
# Accuracy : overall but misleading due to imbalanced dataset
# PRINTS BELOW

def display_classification():
    print(metrics.classification_report(y_test, y_pred))

def display_confusion():
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred).plot()
    plt.show()

# 854 legit correctly identified
# 266 spam correctly caught
# 2 legit emails flagged as spam, wrong
# 24 spam emails slipped through, False Negatives (NEED TO IMPROVE)