import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("IMDB Dataset.csv") 

# Preprocessing
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
X = df['review']
y = df['sentiment']

# Text Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
import pickle
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')