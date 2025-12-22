# NLP Sentiment Analysis (toy)

This repository contains a small script `nlp_sentiment_analysis.py` that trains a logistic regression model on a binary subset of the 20 Newsgroups dataset (used as a proxy for sentiment between `rec.autos` and `talk.politics.misc`).

Quick start

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python3 nlp_sentiment_analysis.py
```

Usage

- Train and save a model (default path `models/model.joblib`):

```bash
python3 nlp_sentiment_analysis.py --mode train --model-path models/model.joblib
```

- Predict using a saved model:

```bash
python3 nlp_sentiment_analysis.py --mode predict --model-path models/model.joblib --sample-text "I love this car" "Policy is awful"
```

- Lightweight wrapper `predict.sh` (after `chmod +x predict.sh`):

```bash
./predict.sh models/model.joblib "I love this car" "Policy is awful"
```

Notes

- If `models/model.joblib` is not present, run `--mode train` to produce it locally.

