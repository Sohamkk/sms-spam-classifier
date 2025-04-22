# app.py

import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = text.split()  # ← simple tokenizer

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

    y = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(y)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    st.write("Transformed Text:", transformed_sms)

    vector_input = vectorizer.transform([transformed_sms])
    st.write("Vector Shape:", vector_input.shape)

    result = model.predict(vector_input)[0]
    st.write("Model Output:", result)
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # This works if the file was fitted!

    if result == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")