## Phishing Detector

This program takes a user's email and predicts whether it is spam or legit using a Logistic Regression
model trained on TF-IDF vectorized text. 

## Pipeline:
1. Load email dataset from csv file
2. Split data into training (80%) and testing (20%) sets
3. Vectorize text using TF-IDf > Removes common stopwords, lowercase all characters and converts words into numerical representations
4. Train a Logistic Regression model on the training set to learn patterns that seperate spam from legit emails
5. Evaluate on unseen data using precision, recall and F1-Score
6. Predict - Given a new email, vectorize it and run it through the trained model to get a spam or legit prediction with a confidence score

## Libraries Used
**pandas** - loading and exploring the dataset from csv file
**scikit-learn** - ML functions including TF-IDF vectorization, Logistic Regression and evaluation metrics
**matplotlib** - displaying the visualization for confusion matrix

## Results
This ML model has the following scores according to a classification report:
Legit Email:
Precision 97% | Recall 100% | F1-Score 99%
Scam Email:
Precision 99% | Recall 92% | F1-Score 95%
A recall of 92% means the model successfully catches 92% of real spam from the test
dataset, this means only 8% slips through. 

## Limitations
The model was trained on phishing emails where scam intent is not obvious, instead it is disguised. 
As a result, emails containing obvious language such as "this a a scam" may score lower confidence than
expected since those words carry fairly little weight in the learned vocabulary. 

Confidence scores also tend to be lower on shorter emails due to less text for TF-IDF to analyze.

Note: the dataset this model was trained on is a few years old. Newer phishing emails may use different
language patterns that the model has not been exposed to and may not recognize as accurately. 

## How to Run
1. Clone repository
2. Install dependencies: 'pip install -r requirements.txt'
3. Run 'python train.py' to train and save the model
4. Run 'python main.py' to launch the menu

