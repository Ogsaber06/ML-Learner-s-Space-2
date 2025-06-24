
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|@\w+|#\w+", "", text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

def load_word2vec_model(path='GoogleNews-vectors-negative300.bin'):
    return KeyedVectors.load_word2vec_format(path, binary=True)

def vectorize_tweet(model, tokens):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def sentiment_classification(df, w2v_model):
    df['tokens'] = df['text'].apply(preprocess_text)
    df['vectors'] = df['tokens'].apply(lambda x: vectorize_tweet(w2v_model, x))
    X = np.stack(df['vectors'].values)
    y = df['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Sentiment Classifier Accuracy:", accuracy_score(y_test, preds))
    return clf

def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess_text(tweet)
    vector = vectorize_tweet(w2v_model, tokens).reshape(1, -1)
    return model.predict(vector)[0]
