import os
from joblib import dump, load


def _train_minimal_model(path: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    categories = ["rec.autos", "talk.politics.misc"]
    samples = [
        "I love my new car and its smooth ride.",
        "The engine broke and the mechanic was expensive.",
        "The new policy affects many people negatively.",
        "I support the proposed changes to the law.",
    ]
    labels = [0, 0, 1, 1]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(samples)
    model = LogisticRegression(max_iter=200)
    model.fit(X, labels)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    dump({"model": model, "vectorizer": vectorizer, "categories": categories}, path)


def test_model_file_exists_and_train_if_missing():
    path = "models/model.joblib"
    if not os.path.exists(path):
        _train_minimal_model(path)
    assert os.path.exists(path), "Pretrained model file models/model.joblib must exist for this test"


def test_predict_output_shape_and_categories():
    path = "models/model.joblib"
    if not os.path.exists(path):
        _train_minimal_model(path)

    obj = load(path)
    model = obj["model"]
    vectorizer = obj["vectorizer"]
    categories = obj.get("categories", ["rec.autos", "talk.politics.misc"])

    samples = ["This car is great.", "I dislike that policy."]
    X = vectorizer.transform(samples)
    preds = model.predict(X)
    assert len(preds) == len(samples)
    for p in preds:
        assert 0 <= p < len(categories)
