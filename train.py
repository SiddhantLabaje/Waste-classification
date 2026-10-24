"""Train and evaluate the Smart Waste Classification model."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "smart_waste_classification_500.csv"
MODEL_PATH = PROJECT_DIR / "waste_model.pkl"

FEATURES = [
    "weight_g",
    "moisture_pct",
    "hardness_1_to_10",
    "density_g_cm3",
    "transparency_score",
    "conductivity_score",
]
TARGET = "waste_category"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path=DATASET_PATH):
    """Load the CSV and check that the columns needed by the model are present."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place "
            "'smart_waste_classification_500.csv' next to train.py."
        )

    try:
        data = pd.read_csv(path)
    except pd.errors.ParserError as error:
        raise ValueError(f"Could not parse the dataset CSV: {error}") from error

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError("The dataset has no rows.")

    return data


def inspect_dataset(data):
    """Print the dataset schema, missing-value counts, and target distribution."""
    print("Dataset columns:")
    print(", ".join(data.columns))
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print("\nClass distribution:")
    print(data[TARGET].value_counts(dropna=False).to_string())


def prepare_data(data):
    """Validate target/features and return numeric features and target labels."""
    if data[TARGET].isna().any():
        raise ValueError(f"'{TARGET}' contains missing values; fill or remove them first.")

    labels = data[TARGET].astype(str).str.strip()
    if labels.eq("").any():
        raise ValueError(f"'{TARGET}' contains blank category names.")

    features = data[FEATURES].copy()
    try:
        features = features.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("All six input feature columns must contain numeric values.") from error

    return features, labels


def train_model():
    """Train the random forest, print its test results, and save the model bundle."""
    data = load_dataset()
    inspect_dataset(data)
    features, labels = prepare_data(data)

    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two waste categories are needed to train a classifier.")
    if (class_counts < 2).any():
        too_small = class_counts[class_counts < 2].to_dict()
        raise ValueError(f"Each category needs at least two samples for stratification: {too_small}")

    test_rows = int(len(data) * TEST_SIZE)
    train_rows = len(data) - test_rows
    if test_rows < len(class_counts) or train_rows < len(class_counts):
        raise ValueError(
            "There are too few rows for an 80/20 stratified split across all categories."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    class_names = model.classes_.tolist()

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions, labels=class_names)
    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "class_names": class_names,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    try:
        with MODEL_PATH.open("wb") as model_file:
            pickle.dump(bundle, model_file)
    except OSError as error:
        raise OSError(f"Could not save the trained model to {MODEL_PATH}: {error}") from error

    print("\nTest-set evaluation (weighted precision, recall, and F1):")
    for name, score in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {score:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=class_names, columns=class_names).to_string())
    print(f"\nTrained model saved to: {MODEL_PATH}")
    print(
        "\nNote: This dataset contains synthetic samples. These test metrics do not "
        "demonstrate real-world waste classification performance."
    )


if __name__ == "__main__":
    try:
        train_model()
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Training failed: {error}") from error
