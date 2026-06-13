"""Train the OpenWatch NLP text classification model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "raw" / "synthetic_public_signals.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "openwatch_tfidf_logreg.joblib"
DEFAULT_TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
DEFAULT_TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
LABEL_ORDER = ["red", "orange", "green"]


def build_model() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=5000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(
    input_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """Train and persist the model, train split and test split."""
    dataset = pd.read_csv(input_path)
    dataset["clean_text"] = dataset["text"].apply(clean_text)

    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["label"],
    )

    model = build_model()
    model.fit(train_df["clean_text"], train_df["label"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    predictions = model.predict(test_df["clean_text"])
    print(f"Model saved to {model_path}")
    print(f"Train split saved to {train_path}")
    print(f"Test split saved to {test_path}")
    print()
    print(classification_report(test_df["label"], predictions, labels=LABEL_ORDER))

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenWatch NLP model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_PATH)
    args = parser.parse_args()

    train_model(
        input_path=args.input,
        model_path=args.model_output,
        train_path=args.train_output,
        test_path=args.test_output,
    )


if __name__ == "__main__":
    main()
