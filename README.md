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

Notes

- The script downloads the 20 Newsgroups dataset on first run.
- This is a toy example and not a real sentiment model.
