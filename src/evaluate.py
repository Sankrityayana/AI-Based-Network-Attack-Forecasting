from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_FILE = (
    MODEL_DIR / "lstm_attack_forecasting.keras"
)


# ============================================================
# Load test data
# ============================================================

def load_test_data():

    X_test = np.load(
        DATA_DIR / "X_test_scaled.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return X_test, y_test


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LSTM TEST SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    print("\nTest data:")
    print(
        f"X_test: {X_test.shape}"
    )
    print(
        f"y_test: {y_test.shape}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading trained LSTM..."
    )

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    print(
        f"Model loaded from:\n{MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    probabilities = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    # Default classification threshold
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LSTM TEST RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy:           {accuracy:.4f}"
    )

    print(
        f"Precision:          {precision:.4f}"
    )

    print(
        f"Recall:             {recall:.4f}"
    )

    print(
        f"F1 Score:           {f1:.4f}"
    )

    print(
        f"ROC-AUC:            {roc_auc:.4f}"
    )

    print(
        f"False Positive Rate:{false_positive_rate:.4f}"
    )

    print(
        f"False Negative Rate:{false_negative_rate:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        cm
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Benign",
                "Malicious",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Probability information
    # --------------------------------------------------------

    print("\nPrediction probability statistics:")

    print(
        f"Minimum: {probabilities.min():.6f}"
    )

    print(
        f"Maximum: {probabilities.max():.6f}"
    )

    print(
        f"Mean:    {probabilities.mean():.6f}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_file = (
        RESULTS_DIR
        / "lstm_metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "LSTM NETWORK ATTACK FORECASTING\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Precision: {precision:.6f}\n"
        )

        file.write(
            f"Recall: {recall:.6f}\n"
        )

        file.write(
            f"F1: {f1:.6f}\n"
        )

        file.write(
            f"ROC-AUC: {roc_auc:.6f}\n"
        )

        file.write(
            f"False Positive Rate: "
            f"{false_positive_rate:.6f}\n"
        )

        file.write(
            f"False Negative Rate: "
            f"{false_negative_rate:.6f}\n"
        )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

    print(
        f"\nMetrics saved to:\n"
        f"{metrics_file}"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()