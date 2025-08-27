# nlp-sentiment-analysis
Natural language processing project to classify movie review sentiments using logistic regression and text vectorization.
# sentiment_analysis.py
# Sentiment analysis of text data using NLP preprocessing and logistic regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example: Load a dataset (replace 'reviews.csv' with your actual reviews file)
# The CSV should have columns 'review' (text) and 'sentiment' (1=positive, 0=negative)
data = pd.read_csv('reviews.csv')

# NLP preprocessing and feature extraction
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.2f}")

# Predict sentiment of a custom review (optional)
sample_review = ["I really loved this movie!"]
sample_vec = vectorizer.transform(sample_review)
sample_pred = clf.predict(sample_vec)
print(f"Sentiment of sample review: {'Positive' if sample_pred[0]==1 else 'Negative'}")
