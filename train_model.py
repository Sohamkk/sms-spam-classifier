import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download necessary datasets
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = text.split()  # Tokenize text
    y = []

    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))  # Stemming

    return " ".join(y)

# Load data (Replace 'spam.csv' with your dataset path)
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # 0 = ham, 1 = spam
df['transformed'] = df['text'].apply(transform_text)  # Apply preprocessing

# Train-test split
cv = CountVectorizer()
X = cv.fit_transform(df['transformed'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))