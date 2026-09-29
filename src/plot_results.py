from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

import joblib


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    X_test = np.load(DATA_DIR / "X_test.npy")
    X_test_scaled = np.load(DATA_DIR / "X_test_scaled.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    logistic_model = joblib.load(
        MODEL_DIR / "logistic_regression_baseline.joblib"
    )

    lstm_model = tf.keras.models.load_model(
        MODEL_DIR / "lstm_attack_forecasting.keras"
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    logistic_prob = logistic_model.predict_proba(
        X_test_flat
    )[:, 1]

    lstm_prob = lstm_model.predict(
        X_test_scaled,
        verbose=0
    ).ravel()

    logistic_pred = (logistic_prob >= 0.5).astype(int)
    lstm_pred = (lstm_prob >= 0.5).astype(int)

    # ---------------------------------------------------------
    # 1. Confusion Matrix — Logistic Regression
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        logistic_pred
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("Logistic Regression - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Benign", "Malicious"])
    plt.yticks([0, 1], ["Benign", "Malicious"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix_logistic.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------------------------
    # 2. Confusion Matrix — LSTM
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        lstm_pred
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("LSTM - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Benign", "Malicious"])
    plt.yticks([0, 1], ["Benign", "Malicious"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix_lstm.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------------------------
    # 3. ROC Curve
    # ---------------------------------------------------------

    fpr_logistic, tpr_logistic, _ = roc_curve(
        y_test,
        logistic_prob
    )

    fpr_lstm, tpr_lstm, _ = roc_curve(
        y_test,
        lstm_prob
    )

    auc_logistic = auc(
        fpr_logistic,
        tpr_logistic
    )

    auc_lstm = auc(
        fpr_lstm,
        tpr_lstm
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr_logistic,
        tpr_logistic,
        label=f"Logistic Regression (AUC={auc_logistic:.4f})"
    )

    plt.plot(
        fpr_lstm,
        tpr_lstm,
        label=f"LSTM (AUC={auc_lstm:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Model Comparison")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "roc_curve_comparison.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------------------------
    # 4. Metric Comparison
    # ---------------------------------------------------------

    models = [
        "Logistic Regression",
        "LSTM"
    ]

    accuracy = [
        0.9927,
        0.9964
    ]

    precision = [
        0.9853,
        0.9926
    ]

    recall = [
        1.0000,
        1.0000
    ]

    f1 = [
        0.9926,
        0.9963
    ]

    x = np.arange(len(models))
    width = 0.18

    plt.figure(figsize=(9, 6))

    plt.bar(
        x - 1.5 * width,
        accuracy,
        width,
        label="Accuracy"
    )

    plt.bar(
        x - 0.5 * width,
        precision,
        width,
        label="Precision"
    )

    plt.bar(
        x + 0.5 * width,
        recall,
        width,
        label="Recall"
    )

    plt.bar(
        x + 1.5 * width,
        f1,
        width,
        label="F1"
    )

    plt.xticks(
        x,
        models
    )

    plt.ylim(
        0.97,
        1.005
    )

    plt.ylabel("Score")
    plt.title("Model Performance Comparison")

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "model_performance_comparison.png",
        dpi=200
    )

    plt.close()

    print("=" * 70)
    print("RESULT VISUALIZATION COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        RESULTS_DIR /
        "confusion_matrix_logistic.png"
    )

    print(
        RESULTS_DIR /
        "confusion_matrix_lstm.png"
    )

    print(
        RESULTS_DIR /
        "roc_curve_comparison.png"
    )

    print(
        RESULTS_DIR /
        "model_performance_comparison.png"
    )


if __name__ == "__main__":
    main()