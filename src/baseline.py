from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_FILE = (
    MODEL_DIR / "logistic_regression_baseline.joblib"
)


def load_data():

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


def flatten_sequences(X):

    return X.reshape(
        X.shape[0],
        -1,
    )


def evaluate_model(
    model,
    X,
    y,
    dataset_name,
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y,
        probabilities,
    )

    cm = confusion_matrix(
        y,
        predictions,
    )

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} RESULTS")
    print("=" * 70)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Benign",
                "Malicious",
            ],
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def main():

    print("=" * 70)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_data()

    print("\nOriginal shapes:")

    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")

    X_train = flatten_sequences(X_train)
    X_val = flatten_sequences(X_val)
    X_test = flatten_sequences(X_test)

    print("\nFlattened shapes:")

    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")

    # --------------------------------------------------------
    # Scaled Logistic Regression
    # --------------------------------------------------------

    print(
        "\nTraining scaled Logistic Regression..."
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation",
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\nModel saved to:\n{MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_file = (
        RESULTS_DIR
        / "baseline_metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "LOGISTIC REGRESSION BASELINE\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            "Validation Results\n"
        )

        for key, value in validation_results.items():

            file.write(
                f"{key}: {value:.4f}\n"
            )

        file.write(
            "\nTest Results\n"
        )

        for key, value in test_results.items():

            file.write(
                f"{key}: {value:.4f}\n"
            )

    print(
        f"Metrics saved to:\n{metrics_file}"
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()