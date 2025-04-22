import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = text.split()
    y = []
    for word in text:
        if word.isalnum() and word not in stop_words and word not in string.punctuation:
            y.append(ps.stem(word))
    return " ".join(y)

# Load and clean data
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['transformed'] = df['text'].apply(transform_text)

# ✅ Fit the TF-IDF vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed'])  # <--- THIS IS IMPORTANT
y = df['label']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the fitted model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")
print("📈 Accuracy:", accuracy_score(y_test, model.predict(X_test)))
