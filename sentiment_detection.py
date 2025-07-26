!pip install joblib
!pip install nltk

import streamlit as st
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import re

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")

tfidf_vectorizer_sentiment = joblib.load("Tfidf_vectorizer_sentiment.joblib")
lr_sentiment_model = joblib.load("logistic_regression_model_sentiment.joblib")


lemmatizer = WordNetLemmatizer()


def cleaning(text):
    text = text.lower()

    text = re.sub(r"\d+", '', text)
    word_tokens = word_tokenize(text)

    words_without_punctuations = [word for word in word_tokens if word not in string.punctuation]

    words_without_stopwords = [word for word in words_without_punctuations if word not in stopwords.words("english")]

    lemmatized_words = [lemmatizer.lemmatize(word, pos = "v") for word in words_without_stopwords]

    return " ".join(lemmatized_words)



st.title("Text Sentiment Detector Application")
st.write("This is a Text Sentiment detection application which can help you detect the type of sentiment (Positive, Negative or Neutral) from your text")
user_input = st.text_input("Please enter the text")
 

cleaned_user_input = cleaning(user_input)
tokenized_form = tfidf_vectorizer_sentiment.transform([cleaned_user_input])

make_prediction = st.button("Predict")
if make_prediction:
    prediction = lr_sentiment_model.predict(tokenized_form)[0]
    prediction_prob = lr_sentiment_model.predict_proba(tokenized_form)[0]

    class_labels = {0: "Positive", 1: "Negative", 2: "Neutral"}

    st.write(f"The predicted class of your text is {class_labels[prediction]}")

    st.bar_chart({class_labels[i]: prob for i, prob in enumerate(prediction_prob)})




