#!/usr/bin/env bash
# Lightweight wrapper to run a prediction with the saved model
set -e
MODEL_PATH=${1:-models/model.joblib}
shift || true
if [ $# -eq 0 ]; then
  echo "Usage: ./predict.sh [model-path] <text...>"
  echo "Example: ./predict.sh models/model.joblib \"I love my car\" \"Policy is bad\""
  exit 1
fi
python3 nlp_sentiment_analysis.py --mode predict --model-path "$MODEL_PATH" --sample-text "$@"
