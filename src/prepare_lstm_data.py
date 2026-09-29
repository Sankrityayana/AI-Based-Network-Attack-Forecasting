from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")


SCALER_FILE = (
    MODEL_DIR / "lstm_feature_scaler.joblib"
)


# ============================================================
# Load data
# ============================================================

def load_data():

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# Scale 3D LSTM data
# ============================================================

def scale_sequences(
    scaler,
    X,
):
    """
    Scale 3D LSTM input.

    Input shape:
        (samples, timesteps, features)

    StandardScaler expects:
        (samples, features)

    Therefore we temporarily reshape the data.
    """

    samples, timesteps, features = X.shape

    X_2d = X.reshape(
        samples * timesteps,
        features,
    )

    X_scaled = scaler.transform(
        X_2d
    )

    return X_scaled.reshape(
        samples,
        timesteps,
        features,
    ).astype(np.float32)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LSTM FEATURE SCALING")
    print("=" * 70)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_data()

    print("\nOriginal shapes:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    # --------------------------------------------------------
    # Create scaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    samples, timesteps, features = (
        X_train.shape
    )

    # Flatten ONLY training data for fitting
    X_train_2d = X_train.reshape(
        samples * timesteps,
        features,
    )

    print(
        "\nFitting StandardScaler "
        "using training data only..."
    )

    scaler.fit(
        X_train_2d
    )

    # --------------------------------------------------------
    # Transform all datasets
    # --------------------------------------------------------

    X_train_scaled = scale_sequences(
        scaler,
        X_train,
    )

    X_val_scaled = scale_sequences(
        scaler,
        X_val,
    )

    X_test_scaled = scale_sequences(
        scaler,
        X_test,
    )

    # --------------------------------------------------------
    # Save scaled data
    # --------------------------------------------------------

    np.save(
        DATA_DIR / "X_train_scaled.npy",
        X_train_scaled,
    )

    np.save(
        DATA_DIR / "X_val_scaled.npy",
        X_val_scaled,
    )

    np.save(
        DATA_DIR / "X_test_scaled.npy",
        X_test_scaled,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nScaled shapes:")

    print(
        f"X_train_scaled: "
        f"{X_train_scaled.shape}"
    )

    print(
        f"X_val_scaled:   "
        f"{X_val_scaled.shape}"
    )

    print(
        f"X_test_scaled:  "
        f"{X_test_scaled.shape}"
    )

    print("\nScaled training statistics:")

    print(
        f"Mean: {X_train_scaled.mean():.6f}"
    )

    print(
        f"Std:  {X_train_scaled.std():.6f}"
    )

    print("\nSaved:")

    print(
        "  data/processed/X_train_scaled.npy"
    )

    print(
        "  data/processed/X_val_scaled.npy"
    )

    print(
        "  data/processed/X_test_scaled.npy"
    )

    print(
        f"  {SCALER_FILE}"
    )

    print("\n" + "=" * 70)
    print("LSTM DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()