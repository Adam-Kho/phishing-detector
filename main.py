import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# for Naive Bayes
#from sklearn.naive_bayes import MultinomialNB

# for Logistic Regression
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



df = pd.read_csv('emails.csv')

print(df.head()) # show first 5 rows
print(df.shape)  # show (rows, columns)
print(df['spam'].value_counts()) # 0 = legit | 1 = spam

X = df['text'] # input data, text
y = df['spam'] # answers, 0 or 1
# x is like the questions and y is the answers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# text_size is .2 meaning 20% test, 80% train

# confirm sizes of splits
print(X_train.shape)
print(X_test.shape)

# can improve this with parameters
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, token_pattern=r'[a-zA-Z]+')
X_train_tfidf = vectorizer.fit_transform(X_train) # learn vocabulary AND convert to numbers
X_test_tfidf = vectorizer.transform(X_test) # convert only, vocabulary already learned

print(X_train_tfidf.shape)

model = LogisticRegression()

# model.fit trains the model, looks at TF-IDF numbers alongide the correct labels
# and learns the patterns that seperate spam vs legit
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Reading Classification Report
# Columns: precision (correct when flagged) | recall (caught of all actual) | f1-score (balance of both) | support (actual count in test set)
# Row 0 (legit) : how well model handles legit emails
# Row 1 (spam) : how well model handles spam emails
# Accuracy : overall but misleading due to imbalanced dataset
# PRINTS BELOW
print(metrics.classification_report(y_test, y_pred))


def predict_email(email_text):
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)
    probability = model.predict_proba(email_tfidf)

    if prediction[0] == 1:
        print(f"Spam | {probability[0][1]:.0%} confidence.")
    else:
        print(f"Legit | {probability[0][0]:.0%} confidence.")


predict_email("Congratulations! You have won a free prize. Click here right now to claim your reward!\n")
predict_email("CLICK HERE RIGHT NOW TO WIN A FREE CAR. RIGHT NOW. DON'T WANT TO MISS OUT ON A CHANCE TO WIN A CAR!\n")
predict_email("Hello, I hope you are doing well. Let me know when you are free to book an appointment to more forward with our proposed deal\n")