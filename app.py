"""Streamlit dashboard for Smart Waste Classification."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from train import DATASET_PATH, FEATURES, MODEL_PATH, load_dataset


def load_model_bundle(path=MODEL_PATH):
    """Load the model and its evaluation results saved by train.py."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run 'python train.py' first."
        )

    try:
        with path.open("rb") as model_file:
            bundle = pickle.load(model_file)
    except (pickle.PickleError, EOFError, AttributeError, ImportError, OSError) as error:
        raise ValueError(f"Could not read the trained model file: {error}") from error

    required_keys = {
        "model",
        "features",
        "class_names",
        "metrics",
        "confusion_matrix",
        "random_state",
        "test_size",
    }
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("The model file is incomplete. Run 'python train.py' again.")
    if bundle["features"] != FEATURES:
        raise ValueError("The model file uses unexpected input features. Run 'python train.py' again.")

    return bundle


def validate_input(values):
    """Reject non-finite values and numbers outside the input feature ranges."""
    ranges = {
        "weight_g": (0, None),
        "moisture_pct": (0, 100),
        "hardness_1_to_10": (1, 10),
        "density_g_cm3": (0, None),
        "transparency_score": (0, 1),
        "conductivity_score": (0, 1),
    }

    for feature in FEATURES:
        value = values[feature]
        if not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")
        minimum, maximum = ranges[feature]
        if value < minimum or (maximum is not None and value > maximum):
            if maximum is None:
                expected = f"at least {minimum}"
            else:
                expected = f"between {minimum} and {maximum}"
            raise ValueError(f"{feature} must be {expected}.")


def main():
    st.set_page_config(page_title="Smart Waste Classification", page_icon="♻️")
    st.title("Smart Waste Classification")
    st.write(
        "Enter six measurable characteristics to classify a waste sample using a "
        "Random Forest model."
    )
    st.warning(
        "This project uses 500 synthetic samples. The test metrics shown here do not "
        "demonstrate real-world waste classification performance."
    )

    try:
        data = load_dataset(DATASET_PATH)
        bundle = load_model_bundle(MODEL_PATH)
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(str(error))
        st.stop()

    st.subheader("Dataset class distribution")
    class_counts = data["waste_category"].value_counts().rename_axis("Category")
    st.bar_chart(class_counts)
    st.dataframe(class_counts.rename("Samples"), width="stretch")

    st.subheader("Model evaluation")
    st.caption(
        "Results use a stratified 80/20 train/test split with random seed "
        f"{bundle['random_state']}. Precision, recall, and F1 are weighted averages."
    )
    metric_columns = st.columns(4)
    metric_labels = [
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision"),
        ("recall_weighted", "Recall"),
        ("f1_weighted", "F1-score"),
    ]
    for column, (key, label) in zip(metric_columns, metric_labels):
        column.metric(label, f"{bundle['metrics'][key]:.3f}")

    st.write("Confusion matrix (rows = actual category, columns = predicted category):")
    confusion = pd.DataFrame(
        bundle["confusion_matrix"],
        index=bundle["class_names"],
        columns=bundle["class_names"],
    )
    st.dataframe(confusion, width="stretch")

    st.subheader("Predict a waste category")
    medians = data[FEATURES].median()
    with st.form("prediction_form"):
        values = {
            "weight_g": st.number_input(
                "Weight (g)", min_value=0.0, value=float(medians["weight_g"]), format="%.2f"
            ),
            "moisture_pct": st.number_input(
                "Moisture (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(medians["moisture_pct"]),
                format="%.2f",
            ),
            "hardness_1_to_10": st.number_input(
                "Hardness (1 to 10)",
                min_value=1.0,
                max_value=10.0,
                value=float(medians["hardness_1_to_10"]),
                format="%.1f",
            ),
            "density_g_cm3": st.number_input(
                "Density (g/cm³)",
                min_value=0.0,
                value=float(medians["density_g_cm3"]),
                format="%.3f",
            ),
            "transparency_score": st.number_input(
                "Transparency score (0 to 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(medians["transparency_score"]),
                format="%.3f",
            ),
            "conductivity_score": st.number_input(
                "Conductivity score (0 to 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(medians["conductivity_score"]),
                format="%.3f",
            ),
        }
        submitted = st.form_submit_button("Predict Waste")

    if submitted:
        try:
            validate_input(values)
        except ValueError as error:
            st.error(str(error))
        else:
            try:
                sample = pd.DataFrame([values], columns=FEATURES)
                prediction = bundle["model"].predict(sample)[0]
            except (ValueError, TypeError, AttributeError) as error:
                st.error(f"Prediction failed: {error}")
            else:
                st.success(f"Predicted waste category: **{prediction}**")


if __name__ == "__main__":
    main()
