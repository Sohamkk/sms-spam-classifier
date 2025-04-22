# train_model.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Sample data — replace with real dataset
texts = [
    "Win a free iPhone now!",
    "Hi, how are you doing today?",
    "Congratulations, you've won a lottery!",
    "Are you coming to the party?",
    "URGENT! Your account has been compromised."
]
labels = [1, 0, 1, 0, 1]  # 1 = spam, 0 = ham

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Model
model = RandomForestClassifier()
model.fit(X, labels)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
