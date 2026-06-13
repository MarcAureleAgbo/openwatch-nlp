"""Prediction helpers and CLI for OpenWatch NLP."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "openwatch_tfidf_logreg.joblib"


def predict_text(text: str, model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Predict the risk class for one text and return probabilities."""
    model = joblib.load(model_path)
    cleaned = clean_text(text)
    predicted_label = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    return {
        "text": text,
        "clean_text": cleaned,
        "predicted_label": predicted_label,
        "probabilities": class_probabilities,
    }


def prediction_table(result: dict) -> pd.DataFrame:
    """Format prediction probabilities as a sorted table."""
    rows = [
        {"label": label, "probability": probability}
        for label, probability in result["probabilities"].items()
    ]
    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one OpenWatch NLP text.")
    parser.add_argument("text", type=str, help="Text to classify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    result = predict_text(args.text, model_path=args.model)
    print(f"Predicted label: {result['predicted_label']}")
    print(prediction_table(result).to_string(index=False))


if __name__ == "__main__":
    main()
