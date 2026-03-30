# Adam

import train as t

# confirm sizes of splits
# print(X_train.shape)
# print(X_test.shape)


def predict_email(userEmail):
    email_tfidf = t.vectorizer.transform([userEmail]) # vectorizer hold learned vocab
    prediction = t.model.predict(email_tfidf) # holds the learned weights, without you can't make predictions
    probability = t.model.predict_proba(email_tfidf)

    if prediction[0] == 1:
        print(f"Spam | {probability[0][1]:.0%} confident.")
    else:
        print(f"Legit | {probability[0][0]:.0%} confident.")