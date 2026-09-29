from pathlib import Path

import pandas as pd
import numpy as np


DATA_FILE = Path("data/processed/temporal_network_states.csv")


def main():
    print("=" * 70)
    print("TEMPORAL DATASET SANITY CHECK")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    # ---------------------------------------------------------
    # 1. Check whether target columns are present as features
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. TARGET / FEATURE CHECK")
    print("-" * 70)

    target_columns = [
        "future_malicious",
        "malicious_flow_count",
        "malicious_flow_ratio",
    ]

    for column in target_columns:
        if column in df.columns:
            print(f"[FOUND] {column}")
        else:
            print(f"[NOT FOUND] {column}")

    # ---------------------------------------------------------
    # 2. Check the actual model feature list
    # ---------------------------------------------------------
    model_features = [
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

    print("\nModel features:")
    for feature in model_features:
        print(f" - {feature}")

    forbidden = [
        feature
        for feature in model_features
        if feature in target_columns
    ]

    print("\nDirect target leakage check:")

    if forbidden:
        print("WARNING: Target-related columns found:")
        for feature in forbidden:
            print(f"  {feature}")
    else:
        print("PASS: No direct target columns are used as model features.")

    # ---------------------------------------------------------
    # 3. Check correlation with target
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. FEATURE CORRELATION WITH FUTURE TARGET")
    print("-" * 70)

    correlations = []

    for feature in model_features:
        if feature in df.columns:
            correlation = df[feature].corr(df["future_malicious"])
            correlations.append((feature, correlation))

    correlations.sort(
        key=lambda x: abs(x[1]) if pd.notna(x[1]) else 0,
        reverse=True,
    )

    for feature, correlation in correlations:
        print(f"{feature:35s}: {correlation: .4f}")

    # ---------------------------------------------------------
    # 4. Target distribution by capture
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. TARGET DISTRIBUTION BY CAPTURE")
    print("-" * 70)

    capture_stats = (
        df.groupby("capture")["future_malicious"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    capture_stats["malicious_percentage"] = (
        capture_stats["mean"] * 100
    )

    print(capture_stats.to_string(index=False))

    # ---------------------------------------------------------
    # 5. Check feature ranges
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. FEATURE RANGES")
    print("-" * 70)

    for feature in model_features:
        if feature in df.columns:
            print(
                f"{feature:35s} "
                f"min={df[feature].min():.4f} "
                f"max={df[feature].max():.4f} "
                f"mean={df[feature].mean():.4f}"
            )

    # ---------------------------------------------------------
    # 6. Check whether captures are strongly separated
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. CAPTURE INFORMATION")
    print("-" * 70)

    for capture, group in df.groupby("capture"):
        print(
            f"{capture:20s} "
            f"windows={len(group):5d} "
            f"malicious_target={group['future_malicious'].mean() * 100:6.2f}%"
        )

    print("\n" + "=" * 70)
    print("SANITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()