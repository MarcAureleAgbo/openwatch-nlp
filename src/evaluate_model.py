"""Evaluate the trained OpenWatch NLP classifier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "openwatch_tfidf_logreg.joblib"
DEFAULT_TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "classification_report.txt"
DEFAULT_METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
DEFAULT_CONFUSION_MATRIX_CSV_PATH = ROOT_DIR / "reports" / "confusion_matrix.csv"
DEFAULT_CONFUSION_MATRIX_FIGURE_PATH = ROOT_DIR / "reports" / "figures" / "confusion_matrix.png"
LABEL_ORDER = ["red", "orange", "green"]


def evaluate_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    confusion_matrix_csv_path: Path = DEFAULT_CONFUSION_MATRIX_CSV_PATH,
    confusion_matrix_figure_path: Path = DEFAULT_CONFUSION_MATRIX_FIGURE_PATH,
) -> dict:
    """Evaluate the persisted model and write text, JSON and figure outputs."""
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    predictions = model.predict(test_df["clean_text"])
    report_text = classification_report(test_df["label"], predictions, labels=LABEL_ORDER)
    report_dict = classification_report(
        test_df["label"],
        predictions,
        labels=LABEL_ORDER,
        output_dict=True,
    )
    matrix = confusion_matrix(test_df["label"], predictions, labels=LABEL_ORDER)
    matrix_df = pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_csv_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_figure_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(report_text, encoding="utf-8")
    metrics_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    matrix_df.to_csv(confusion_matrix_csv_path)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_ORDER)
    display.plot(cmap="Blues", values_format="d")
    plt.title("OpenWatch NLP - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_figure_path, dpi=160)
    plt.close()

    print(report_text)
    print(f"Classification report saved to {report_path}")
    print(f"Metrics JSON saved to {metrics_path}")
    print(f"Confusion matrix CSV saved to {confusion_matrix_csv_path}")
    print(f"Confusion matrix figure saved to {confusion_matrix_figure_path}")

    return report_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenWatch NLP model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    args = parser.parse_args()

    evaluate_model(model_path=args.model, test_path=args.test)


if __name__ == "__main__":
    main()
