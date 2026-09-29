from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_FILE = MODEL_DIR / "lstm_attack_forecasting.keras"
HISTORY_FILE = RESULTS_DIR / "lstm_training_history.json"

SEQUENCE_LENGTH = 5
N_FEATURES = 13

EPOCHS = 30
BATCH_SIZE = 64
RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        DATA_DIR / "X_train_scaled.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_val_scaled.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    )

    return X_train, y_train, X_val, y_val


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(SEQUENCE_LENGTH, N_FEATURES)
            ),

            tf.keras.layers.LSTM(
                64
            ),

            tf.keras.layers.Dropout(
                0.20
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid"
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("LSTM NETWORK ATTACK FORECASTING")
    print("=" * 70)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading training data...")

    X_train, y_train, X_val, y_val = load_data()

    print("\nDataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"y_val:   {y_val.shape}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding LSTM model...")

    model = build_model()

    print("\nModel architecture:")
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            checkpoint,
            early_stopping
        ],

        verbose=1
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    trained_epochs = len(
        history.history["loss"]
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nEpochs actually trained: "
        f"{trained_epochs}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.6f}"
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )

    print(
        f"\nTraining history saved to:"
        f"\n{HISTORY_FILE}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()