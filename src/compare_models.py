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
)


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

BASELINE_MODEL = MODEL_DIR / "logistic_regression_baseline.joblib"
LSTM_MODEL = MODEL_DIR / "lstm_attack_forecasting.keras"


def evaluate_model(model_name, model, X_test, y_test, is_lstm=False):
    if is_lstm:
        probabilities = model.predict(X_test, verbose=0).ravel()
    else:
        probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():

    print("=" * 70)
    print("LOGISTIC REGRESSION vs LSTM")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    X_test = np.load(DATA_DIR / "X_test.npy")
    X_test_scaled = np.load(DATA_DIR / "X_test_scaled.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    print("\nTest dataset:")
    print(f"X_test:        {X_test.shape}")
    print(f"X_test_scaled: {X_test_scaled.shape}")
    print(f"y_test:        {y_test.shape}")

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    print("\nLoading Logistic Regression...")

    import joblib

    baseline = joblib.load(BASELINE_MODEL)

    baseline_result = evaluate_model(
        "Logistic Regression",
        baseline,
        X_test.reshape(X_test.shape[0], -1),
        y_test,
        is_lstm=False,
    )

    # ---------------------------------------------------------
    # LSTM
    # ---------------------------------------------------------

    print("Loading LSTM...")

    lstm = tf.keras.models.load_model(LSTM_MODEL)

    lstm_result = evaluate_model(
        "LSTM",
        lstm,
        X_test_scaled,
        y_test,
        is_lstm=True,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results = [
        baseline_result,
        lstm_result,
    ]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    header = (
        f"{'Model':<22}"
        f"{'Accuracy':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'ROC-AUC':>10}"
        f"{'FPR':>10}"
    )

    print("\n" + header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['Model']:<22}"
            f"{result['Accuracy']:>10.4f}"
            f"{result['Precision']:>12.4f}"
            f"{result['Recall']:>10.4f}"
            f"{result['F1']:>10.4f}"
            f"{result['ROC-AUC']:>10.4f}"
            f"{result['False Positive Rate']:>10.4f}"
        )

    # ---------------------------------------------------------
    # Confusion matrices
    # ---------------------------------------------------------

    print("\nConfusion Matrices:")

    for result in results:
        print(f"\n{result['Model']}:")
        print(
            np.array([
                [result["TN"], result["FP"]],
                [result["FN"], result["TP"]],
            ])
        )

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "model_comparison.txt"

    with open(output_file, "w", encoding="utf-8") as file:

        file.write("NETWORK ATTACK FORECASTING MODEL COMPARISON\n")
        file.write("=" * 60 + "\n\n")

        for result in results:

            file.write(f"Model: {result['Model']}\n")
            file.write("-" * 40 + "\n")

            file.write(
                f"Accuracy: {result['Accuracy']:.6f}\n"
            )
            file.write(
                f"Precision: {result['Precision']:.6f}\n"
            )
            file.write(
                f"Recall: {result['Recall']:.6f}\n"
            )
            file.write(
                f"F1: {result['F1']:.6f}\n"
            )
            file.write(
                f"ROC-AUC: {result['ROC-AUC']:.6f}\n"
            )
            file.write(
                f"False Positive Rate: "
                f"{result['False Positive Rate']:.6f}\n"
            )
            file.write(
                f"False Negative Rate: "
                f"{result['False Negative Rate']:.6f}\n"
            )

            file.write("\nConfusion Matrix:\n")

            file.write(
                f"[[{result['TN']}, {result['FP']}],\n"
                f" [{result['FN']}, {result['TP']}]]\n\n"
            )

    print(f"\nComparison saved to:")
    print(output_file)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()