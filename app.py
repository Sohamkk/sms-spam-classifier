import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

# Download punkt tokenizer
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)  # Tokenize text

    # Remove stopwords and punctuation, then apply stemming
    processed_text = [
        ps.stem(word) for word in text
        if word.isalnum() and word not in stop_words and word not in string.punctuation
    ]

    return " ".join(processed_text)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # Preprocess input text
    transformed_sms = transform_text(input_sms)
    st.write("Transformed Text:", transformed_sms)

    # Vectorize the transformed text
    vector_input = vectorizer.transform([transformed_sms])
    st.write("Vector Shape:", vector_input.shape)

    # Predict result
    result = model.predict(vector_input)[0]
    st.write("Model Output:", result)

    # Display prediction results
    if result == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")
