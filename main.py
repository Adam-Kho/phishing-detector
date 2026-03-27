import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# for Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df = pd.read_csv('emails.csv')

print(df.head()) # show first 5 rows
print(df.shape)  # show 5728 emails and 2 columns
print(df['spam'].value_counts()) # 0 = legit | 1 = spam

x = df['text'] # input data, text
y = df['spam'] # answers, 0 or 1
# x is like the questions and y is the answers

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# text_size is .2 meaning 20% test, 80% train

# confirm sizes of splits
print(x_train.shape)
print(x_test.shape)

vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

print(x_train_tfidf.shape)

model = MultinomialNB()
model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)

print(metrics.classification_report(y_test, y_pred))