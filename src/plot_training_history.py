from pathlib import Path
import json

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")

HISTORY_FILE = (
    RESULTS_DIR /
    "lstm_training_history.json"
)


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if not HISTORY_FILE.exists():

        raise FileNotFoundError(
            f"\nTraining history not found:\n"
            f"{HISTORY_FILE}\n\n"
            f"Run 'python src/train.py' first "
            f"so the training history can be saved."
        )

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOSS PLOT
# ============================================================

def create_loss_plot(history):

    epochs = range(
        1,
        len(history["loss"]) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy Loss")

    plt.title(
        "LSTM Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR /
        "lstm_training_loss.png"
    )

    plt.savefig(
        output_file,
        dpi=200
    )

    plt.close()

    return output_file


# ============================================================
# ACCURACY PLOT
# ============================================================

def create_accuracy_plot(history):

    epochs = range(
        1,
        len(history["accuracy"]) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "LSTM Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        RESULTS_DIR /
        "lstm_training_accuracy.png"
    )

    plt.savefig(
        output_file,
        dpi=200
    )

    plt.close()

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LSTM TRAINING HISTORY VISUALIZATION")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history = load_history()

    epochs_count = len(
        history["loss"]
    )

    print(
        f"\nEpochs found: "
        f"{epochs_count}"
    )

    print(
        f"Final training accuracy: "
        f"{history['accuracy'][-1]:.6f}"
    )

    print(
        f"Final validation accuracy: "
        f"{history['val_accuracy'][-1]:.6f}"
    )

    print(
        f"Final training loss: "
        f"{history['loss'][-1]:.8f}"
    )

    print(
        f"Final validation loss: "
        f"{history['val_loss'][-1]:.8f}"
    )

    loss_file = create_loss_plot(
        history
    )

    accuracy_file = create_accuracy_plot(
        history
    )

    print("\nGenerated files:")

    print(
        f" - {loss_file}"
    )

    print(
        f" - {accuracy_file}"
    )

    print("\n" + "=" * 70)
    print("TRAINING HISTORY VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()