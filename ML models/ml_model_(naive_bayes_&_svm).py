import requests
import pandas as pd
url = "https://raw.githubusercontent.com/TranBang875/seminar2project/main/dataset/train.csv"
response = requests.get(url)

with open("train.csv", "wb") as f:
    f.write(response.content)

df = pd.read_csv("train.csv")
display(df.head())

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # 3. Convert the input text to lowercase.
    text = text.lower()
    # 4. Remove non-alphanumeric characters from the text using regular expressions.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # 5. Remove stop words from the text.
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 6. Lemmatize the remaining words in the text.
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    # 7. Apply the emotion cleaning logic to the text (placeholder for now)
    # This step requires specific logic based on how emotions are represented.
    # For this subtask, we'll assume no specific emotion cleaning is needed beyond general text cleaning.
    return text

df['cleaned_comment_text'] = df['comment_text'].apply(clean_text)
display(df[['comment_text', 'cleaned_comment_text']].head())

import nltk
nltk.download('punkt')

import nltk
from nltk.tokenize import word_tokenize

try:
    df['tokenized_comment_text'] = df['cleaned_comment_text'].apply(word_tokenize)
except LookupError:
    nltk.download('punkt_tab')
    df['tokenized_comment_text'] = df['cleaned_comment_text'].apply(word_tokenize)

display(df[['cleaned_comment_text', 'tokenized_comment_text']].head())

display(df[['comment_text', 'cleaned_comment_text', 'tokenized_comment_text']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Join the tokens back into strings for TF-IDF
df['tokenized_comment_text_str'] = df['tokenized_comment_text'].apply(lambda x: ' '.join(x))

tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features
tfidf_features = tfidf_vectorizer.fit_transform(df['tokenized_comment_text_str'])

print("Shape of TF-IDF features:", tfidf_features.shape)

import requests

url = "https://raw.githubusercontent.com/TranBang875/seminar2project/main/dataset/test.csv"
response = requests.get(url)

with open("test.csv", "wb") as f:
    f.write(response.content)

test_df = pd.read_csv("test.csv")
display(test_df.head())

test_df['cleaned_comment_text'] = test_df['comment_text'].apply(clean_text)
test_df['tokenized_comment_text'] = test_df['cleaned_comment_text'].apply(word_tokenize)
display(test_df[['comment_text', 'cleaned_comment_text', 'tokenized_comment_text']].head())

test_df['cleaned_comment_text'] = test_df['comment_text'].apply(clean_text)
test_df['tokenized_comment_text'] = test_df['cleaned_comment_text'].apply(word_tokenize)
display(test_df[['comment_text', 'cleaned_comment_text', 'tokenized_comment_text']].head())

test_df['tokenized_comment_text_str'] = test_df['tokenized_comment_text'].apply(lambda x: ' '.join(x))
tfidf_test_features = tfidf_vectorizer.transform(test_df['tokenized_comment_text_str'])

print("Shape of TF-IDF test features:", tfidf_test_features.shape)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Define the target labels
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

x_train = tfidf_features
y_train = df[LABELS]

# Initialize and train the multi-output logistic regression model
model = MultiOutputClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
model.fit(x_train, y_train)

x_test = tfidf_test_features
y_test = test_df[LABELS]
print("Model training complete.")

from sklearn.metrics import f1_score, classification_report

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate the F1 score for each label
f1_scores = f1_score(y_test, y_pred, average=None)

# Print the F1 scores for each label
for i, label in enumerate(LABELS):
    print(f"F1 score for {label}: {f1_scores[i]:.4f}")

# Calculate the average F1 score
average_f1_score = f1_score(y_test, y_pred, average='weighted')
print(f"\nAverage F1 score (weighted): {average_f1_score:.4f}")

# Print classification report for precision, recall, and support
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS))

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

# Define the target labels
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

x_train = tfidf_features
y_train = df[LABELS]

# Initialize and train the multi-output logistic regression model
model = MultiOutputClassifier(SVC())
model.fit(x_train, y_train)

x_test = tfidf_test_features
y_test = test_df[LABELS]
print("Model training complete.")

from sklearn.metrics import f1_score, classification_report

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate the F1 score for each label
f1_scores = f1_score(y_test, y_pred, average=None)

# Print the F1 scores for each label
for i, label in enumerate(LABELS):
    print(f"F1 score for {label}: {f1_scores[i]:.4f}")

# Calculate the average F1 score
average_f1_score = f1_score(y_test, y_pred, average='weighted')
print(f"\nAverage F1 score (weighted): {average_f1_score:.4f}")

# Print classification report for precision, recall, and support
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS))

"""Naive Bayes"""

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB

# Define the target labels
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

x_train = tfidf_features
y_train = df[LABELS]

# Initialize and train the multi-output logistic regression model
model = MultiOutputClassifier(MultinomialNB())
model.fit(x_train, y_train)

x_test = tfidf_test_features
y_test = test_df[LABELS]
print("Model training complete.")

from sklearn.metrics import f1_score, classification_report, accuracy_score

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate the F1 score for each label
f1_scores = f1_score(y_test, y_pred, average=None)

# Print the F1 scores for each label
for i, label in enumerate(LABELS):
    print(f"F1 score for {label}: {f1_scores[i]:.4f}")

# Calculate the average F1 score
average_f1_score = f1_score(y_test, y_pred, average='weighted')
print(f"\nAverage F1 score (weighted): {average_f1_score:.4f}")

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy score: {accuracy:.4f}")

# Print classification report for precision, recall, and support
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS))

from sklearn.metrics import precision_score, recall_score

# Calculate macro and micro average precision, recall, and f1-score
macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_f1 = f1_score(y_test, y_pred, average='macro')

micro_precision = precision_score(y_test, y_pred, average='micro')
micro_recall = recall_score(y_test, y_pred, average='micro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print(f"\nMacro Average Precision: {macro_precision:.4f}")
print(f"Macro Average Recall: {macro_recall:.4f}")
print(f"Macro Average F1-score: {macro_f1:.4f}")

print(f"\nMicro Average Precision: {micro_precision:.4f}")
print(f"Micro Average Recall: {micro_recall:.4f}")
print(f"Micro Average F1-score: {micro_f1:.4f}")



