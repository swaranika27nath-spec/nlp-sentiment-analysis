import argparse
import os
from joblib import dump, load
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def train_and_save(model_path: str, categories=None, save_vectorizer: bool = False):
    if categories is None:
        categories = ["rec.autos", "talk.politics.misc"]

    data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
    X_text = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=categories))

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    # Default: save both in one joblib file for convenience
    dump({"model": model, "vectorizer": vectorizer, "categories": categories}, model_path)
    print(f"Saved model+vectorizer to: {model_path}")

    if save_vectorizer:
        vec_path = model_path + ".vectorizer.joblib"
        dump(vectorizer, vec_path)
        model_only_path = model_path + ".model-only.joblib"
        dump({"model": model, "categories": categories}, model_only_path)
        print(f"Also saved vectorizer to: {vec_path} and model-only to: {model_only_path}")


def predict_with_model(model_path: str, sample_texts):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    obj = load(model_path)
    model = obj["model"]
    vectorizer = obj["vectorizer"]
    categories = obj.get("categories", ["rec.autos", "talk.politics.misc"])

    X = vectorizer.transform(sample_texts)
    preds = model.predict(X)
    for text, label in zip(sample_texts, preds):
        print(f"Text: {text} -> Predicted label: {categories[label]}")


def parse_args():
    p = argparse.ArgumentParser(description="Train or predict with a simple text classifier")
    p.add_argument("--mode", choices=["train", "predict"], default="train", help="Operation mode")
    p.add_argument("--model-path", default="models/model.joblib", help="Path to save/load the model+vectorizer")
    p.add_argument("--save-vectorizer", action="store_true", help="Also save the vectorizer to a separate file alongside the model")
    p.add_argument("--sample-text", nargs="*", help="Sample text(s) to predict (for predict mode)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_and_save(args.model_path, save_vectorizer=args.save_vectorizer)
    else:
        if not args.sample_text:
            print("Please provide --sample-text when using --mode predict")
            return
        predict_with_model(args.model_path, args.sample_text)


if __name__ == "__main__":
    main()
