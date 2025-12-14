# tests/test_dummy.py

def test_dummy():
    # Simple test that always passes
    assert 1 + 1 == 2 

# nlp_sentiment_analysis.py

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate():
    # 1. Load data
    categories = ["rec.autos", "talk.politics.misc"]
    data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    X_text = data.data
    y = data.target

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Vectorize
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(
        "Classification report:\n",
        classification_report(y_test, y_pred, target_names=categories),
    )

    return model, vectorizer


if __name__ == "__main__":
    train_and_evaluate()

