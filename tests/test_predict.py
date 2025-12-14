import os
from joblib import load


def test_model_file_exists():
    path = "models/model.joblib"
    assert os.path.exists(path), "Pretrained model file models/model.joblib must exist for this test"


def test_predict_output_shape_and_categories():
    obj = load("models/model.joblib")
    model = obj["model"]
    vectorizer = obj["vectorizer"]
    categories = obj.get("categories", ["rec.autos", "talk.politics.misc"])

    samples = ["This car is great.", "I dislike that policy."]
    X = vectorizer.transform(samples)
    preds = model.predict(X)
    assert len(preds) == len(samples)
    for p in preds:
        assert 0 <= p < len(categories)

        import os
from joblib import load

def test_model_file_exists():
    path = "models/model.joblib"
    assert os.path.exists(path), "Pretrained model file models/model.joblib must exist for this test"

def test_predict_output_shape_and_categories():
    obj = load("models/model.joblib")
    model = obj["model"]
    vectorizer = obj["vectorizer"]
    categories = obj.get("categories", ["rec.autos", "talk.politics.misc"])

    samples = ["This car is great.", "I dislike that policy."]
    X = vectorizer.transform(samples)
    preds = model.predict(X)
    assert len(preds) == len(samples)
    for p in preds:
        assert 0 <= p < len(categories)
