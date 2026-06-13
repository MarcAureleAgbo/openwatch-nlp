"""Streamlit application for OpenWatch NLP."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib"))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import altair as alt
import pandas as pd
import streamlit as st

from src.predict import DEFAULT_MODEL_PATH, predict_text, prediction_table


DATA_PATH = ROOT_DIR / "data" / "raw" / "synthetic_public_signals.csv"
REPORT_PATH = ROOT_DIR / "reports" / "classification_report.txt"
CONFUSION_MATRIX_PATH = ROOT_DIR / "reports" / "figures" / "confusion_matrix.png"
LABEL_ORDER = ["red", "orange", "green"]
RISK_COLORS = {
    "red": "#D64545",
    "orange": "#E6952A",
    "green": "#2E9D68",
}
RISK_DISPLAY_NAMES = {
    "red": "Critical",
    "orange": "Intermediate",
    "green": "Non-relevant",
}

EXAMPLE_TEXTS = {
    "Critical signal": "A public post claims that administrator credentials were leaked and attackers are testing access.",
    "Intermediate signal": "Several users report unusual login prompts and repeated password reset emails.",
    "Non-relevant content": "The company announced a dashboard update with minor interface improvements.",
}


def render_metric_cards(dataset: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Texts", f"{len(dataset):,}")
    col2.metric("Red", int((dataset["label"] == "red").sum()))
    col3.metric("Orange", int((dataset["label"] == "orange").sum()))
    col4.metric("Green", int((dataset["label"] == "green").sum()))


def label_color_scale() -> alt.Scale:
    return alt.Scale(domain=LABEL_ORDER, range=[RISK_COLORS[label] for label in LABEL_ORDER])


def render_label_distribution(dataset: pd.DataFrame) -> None:
    counts = (
        dataset["label"]
        .value_counts()
        .reindex(LABEL_ORDER)
        .fillna(0)
        .astype(int)
        .rename_axis("label")
        .reset_index(name="count")
    )
    counts["risk_level"] = counts["label"].map(RISK_DISPLAY_NAMES)

    chart = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("label:N", sort=LABEL_ORDER, title="Risk label"),
            y=alt.Y("count:Q", title="Number of texts"),
            color=alt.Color("label:N", scale=label_color_scale(), legend=None),
            tooltip=[
                alt.Tooltip("risk_level:N", title="Risk level"),
                alt.Tooltip("count:Q", title="Texts"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_prediction_badge(label: str) -> None:
    color = RISK_COLORS[label]
    display_name = RISK_DISPLAY_NAMES[label]
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:0.5rem 0.85rem;
            border-radius:6px;
            background:{color};
            color:white;
            font-weight:700;
            letter-spacing:0;
        ">
            {label.upper()} - {display_name}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probabilities: pd.DataFrame) -> None:
    chart_data = probabilities.copy()
    chart_data["risk_level"] = chart_data["label"].map(RISK_DISPLAY_NAMES)

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("probability:Q", title="Probability", axis=alt.Axis(format="%")),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color("label:N", scale=label_color_scale(), legend=None),
            tooltip=[
                alt.Tooltip("risk_level:N", title="Risk level"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=160)
    )
    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="OpenWatch NLP", layout="wide")

    st.title("OpenWatch NLP")
    st.caption("Synthetic weak-signal classification with TF-IDF and Logistic Regression")

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            "Model not found. Run `python -m src.data_generation`, "
            "`python -m src.train_model`, then `python -m src.evaluate_model` first."
        )
        st.stop()

    tab_predict, tab_dataset, tab_evaluation = st.tabs(["Predict", "Dataset", "Evaluation"])

    with tab_predict:
        st.subheader("Classify a public text signal")
        selected_example = st.selectbox("Example", list(EXAMPLE_TEXTS.keys()))
        default_text = EXAMPLE_TEXTS[selected_example]
        text = st.text_area("Text to classify", value=default_text, height=140)

        if st.button("Classify", type="primary"):
            result = predict_text(text)
            label = result["predicted_label"]
            st.write("Predicted risk level")
            render_prediction_badge(label)

            probabilities = prediction_table(result)
            render_probability_chart(probabilities)
            st.dataframe(
                probabilities.style.format({"probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.progress(float(probabilities.iloc[0]["probability"]))

    with tab_dataset:
        st.subheader("Synthetic dataset")
        if DATA_PATH.exists():
            dataset = pd.read_csv(DATA_PATH)
            render_metric_cards(dataset)
            st.dataframe(dataset.head(20), use_container_width=True, hide_index=True)
            render_label_distribution(dataset)
        else:
            st.warning("Synthetic dataset not found yet.")

    with tab_evaluation:
        st.subheader("Model evaluation")
        if REPORT_PATH.exists():
            st.code(REPORT_PATH.read_text(encoding="utf-8"), language="text")
        else:
            st.warning("Classification report not found yet.")

        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix")
        else:
            st.warning("Confusion matrix figure not found yet.")


if __name__ == "__main__":
    main()
