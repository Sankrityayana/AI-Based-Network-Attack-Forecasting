from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "data/processed/temporal_network_states.csv"
)

OUTPUT_DIR = Path("data/processed")


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 5

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# Features used by the LSTM
# ============================================================

FEATURE_COLUMNS = [
    "flow_count",
    "total_orig_bytes",
    "total_resp_bytes",
    "total_orig_pkts",
    "total_resp_pkts",
    "mean_duration",
    "total_missed_bytes",
    "unique_source_ports",
    "unique_destination_ports",
    "tcp_flow_count",
    "udp_flow_count",
    "successful_connection_count",
    "rejected_connection_count",
]

TARGET_COLUMN = "future_malicious"


# ============================================================
# Create sequences
# ============================================================

def create_sequences(
    df,
    sequence_length=SEQUENCE_LENGTH,
):
    """
    Create sequences from one chronological dataframe.

    The sequence contains the previous `sequence_length`
    windows and predicts the following window.
    """

    features = df[
        FEATURE_COLUMNS
    ].values.astype(np.float32)

    targets = df[
        TARGET_COLUMN
    ].values.astype(np.int32)

    X = []
    y = []

    if len(df) <= sequence_length:
        return (
            np.empty(
                (0, sequence_length, len(FEATURE_COLUMNS)),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.int32,
            ),
        )

    for i in range(
        len(df) - sequence_length
    ):

        X.append(
            features[
                i : i + sequence_length
            ]
        )

        y.append(
            targets[
                i + sequence_length
            ]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
    )


# ============================================================
# Split one capture chronologically
# ============================================================

def split_capture(df):
    """
    Split a capture chronologically into:

        70% train
        15% validation
        15% test
    """

    df = df.sort_values(
        "window_id"
    ).reset_index(drop=True)

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = train_end + int(
        n * VALIDATION_RATIO
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CHRONOLOGICAL LSTM DATASET SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Total temporal windows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    X_train_all = []
    y_train_all = []

    X_val_all = []
    y_val_all = []

    X_test_all = []
    y_test_all = []

    # --------------------------------------------------------
    # Process each capture independently
    # --------------------------------------------------------

    for capture_name, capture_df in df.groupby(
        "capture",
        sort=False,
    ):

        capture_df = capture_df.sort_values(
            "window_id"
        ).reset_index(drop=True)

        print("\n" + "-" * 70)
        print(f"Capture: {capture_name}")
        print(
            f"Total windows: "
            f"{len(capture_df):,}"
        )

        # --------------------------------------------
        # Split chronologically
        # --------------------------------------------

        (
            train_df,
            validation_df,
            test_df,
        ) = split_capture(capture_df)

        print(
            f"Train windows:      {len(train_df):,}"
        )

        print(
            f"Validation windows: {len(validation_df):,}"
        )

        print(
            f"Test windows:       {len(test_df):,}"
        )

        # --------------------------------------------
        # Build sequences independently
        # --------------------------------------------

        X_train, y_train = create_sequences(
            train_df
        )

        X_val, y_val = create_sequences(
            validation_df
        )

        X_test, y_test = create_sequences(
            test_df
        )

        print(
            f"Train sequences:      {len(X_train):,}"
        )

        print(
            f"Validation sequences: {len(X_val):,}"
        )

        print(
            f"Test sequences:       {len(X_test):,}"
        )

        # --------------------------------------------
        # Store
        # --------------------------------------------

        if len(X_train) > 0:
            X_train_all.append(X_train)
            y_train_all.append(y_train)

        if len(X_val) > 0:
            X_val_all.append(X_val)
            y_val_all.append(y_val)

        if len(X_test) > 0:
            X_test_all.append(X_test)
            y_test_all.append(y_test)

    # --------------------------------------------------------
    # Combine captures
    # --------------------------------------------------------

    X_train = np.concatenate(
        X_train_all,
        axis=0,
    )

    y_train = np.concatenate(
        y_train_all,
        axis=0,
    )

    X_val = np.concatenate(
        X_val_all,
        axis=0,
    )

    y_val = np.concatenate(
        y_val_all,
        axis=0,
    )

    X_test = np.concatenate(
        X_test_all,
        axis=0,
    )

    y_test = np.concatenate(
        y_test_all,
        axis=0,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        OUTPUT_DIR / "X_val.npy",
        X_val,
    )

    np.save(
        OUTPUT_DIR / "y_val.npy",
        y_val,
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SEQUENCE DATASET")
    print("=" * 70)

    print("\nShapes:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"y_val:   {y_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )

    print("\nTarget distributions:")

    print(
        "\nTraining:"
    )

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    print(
        "\nValidation:"
    )

    print(
        pd.Series(y_val)
        .value_counts()
        .sort_index()
    )

    print(
        "\nTest:"
    )

    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
    )

    print("\nSaved:")

    print(
        "  data/processed/X_train.npy"
    )

    print(
        "  data/processed/y_train.npy"
    )

    print(
        "  data/processed/X_val.npy"
    )

    print(
        "  data/processed/y_val.npy"
    )

    print(
        "  data/processed/X_test.npy"
    )

    print(
        "  data/processed/y_test.npy"
    )

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()