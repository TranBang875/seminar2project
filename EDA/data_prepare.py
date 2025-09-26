


import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/TranBang875/seminar2project/main/dataset/train.csv")
display(df.head())

"""## Define text cleaning function

### Subtask:
Create a function that applies the following text cleaning steps: removing stop words, lowercasing, retaining alphanumeric characters, lemmatization, and emotion cleaning.

"""

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
    return text

"""## Apply text cleaning

### Subtask:
Apply the text cleaning function to the relevant text column in the DataFrame.

"""

df['cleaned_comment_text'] = df['comment_text'].apply(clean_text)
display(df[['comment_text', 'cleaned_comment_text']].head())

"""## Tokenization

### Subtask:
Tokenize the cleaned text data.

"""

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

