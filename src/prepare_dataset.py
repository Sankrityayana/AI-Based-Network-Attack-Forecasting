from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

RAW_DIR = Path("data/raw/CTU-IDSEVAL-6/zeek/zeek")
OUTPUT_FILE = Path("data/processed/temporal_network_states.csv")

WINDOW_SIZE_SECONDS = 60


# ============================================================
# Dataset columns
# ============================================================

BASE_COLUMNS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
    "label",
    "detailedlabel",
]

BASE_COLUMNS_WITH_IP_PROTO = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
    "ip_proto",
    "label",
    "detailedlabel",
]


# ============================================================
# Helper functions
# ============================================================

def find_log_files():
    """Find all labeled Zeek connection logs."""
    return sorted(RAW_DIR.glob("*.conn-labeled.log"))


def load_conn_log(file_path):
    """
    Load a Zeek conn-labeled log.

    Some CTU-IDSEVAL-6 files contain 23 columns while
    port-scan files contain 24 columns because of ip_proto.
    """

    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.rstrip("\n")

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) == 23:
                columns = BASE_COLUMNS
            elif len(parts) == 24:
                columns = BASE_COLUMNS_WITH_IP_PROTO
            else:
                continue

            if len(parts) != len(columns):
                continue

            rows.append(parts)

    df = pd.DataFrame(rows, columns=columns)

    return df


def clean_flow_data(df):
    """Clean numeric and categorical flow fields."""

    numeric_columns = [
        "ts",
        "id.orig_p",
        "id.resp_p",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Replace Zeek empty/unset values
    df = df.replace(
        {
            "-": pd.NA,
            "(empty)": pd.NA,
            "": pd.NA,
        }
    )

    # Numerical fields
    for column in numeric_columns:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    # Categorical fields
    categorical_columns = [
        "proto",
        "service",
        "conn_state",
        "history",
        "label",
        "detailedlabel",
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna("unknown")

    return df


def create_temporal_windows(df, capture_name):
    """
    Convert flow-level records into 60-second network states.
    """

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("ts").reset_index(drop=True)

    # Relative time starts from zero for each capture.
    first_timestamp = df["ts"].iloc[0]

    df["relative_time"] = df["ts"] - first_timestamp

    df["window_id"] = (
        df["relative_time"] // WINDOW_SIZE_SECONDS
    ).astype(int)

    grouped = df.groupby("window_id", sort=True)

    windows = grouped.agg(
        flow_count=("uid", "count"),

        total_orig_bytes=("orig_bytes", "sum"),
        total_resp_bytes=("resp_bytes", "sum"),

        total_orig_pkts=("orig_pkts", "sum"),
        total_resp_pkts=("resp_pkts", "sum"),

        mean_duration=("duration", "mean"),

        total_missed_bytes=("missed_bytes", "sum"),

        unique_source_ports=("id.orig_p", "nunique"),
        unique_destination_ports=("id.resp_p", "nunique"),

        tcp_flow_count=(
            "proto",
            lambda x: (x == "tcp").sum(),
        ),

        udp_flow_count=(
            "proto",
            lambda x: (x == "udp").sum(),
        ),

        successful_connection_count=(
            "conn_state",
            lambda x: x.isin(["SF", "S1"]).sum(),
        ),

        rejected_connection_count=(
            "conn_state",
            lambda x: x.isin(["REJ", "RSTO", "RSTR"]).sum(),
        ),

        malicious_flow_count=(
            "label",
            lambda x: (x == "Malicious").sum(),
        ),
    ).reset_index()

    # Calculate malicious ratio
    windows["malicious_flow_ratio"] = (
        windows["malicious_flow_count"]
        / windows["flow_count"]
    )

    # --------------------------------------------------------
    # Create future target
    # --------------------------------------------------------

    windows["next_malicious_flow_count"] = (
        windows["malicious_flow_count"].shift(-1)
    )

    # Last window has no known future.
    # Remove it instead of incorrectly assigning target = 0.
    windows = windows[
        windows["next_malicious_flow_count"].notna()
    ].copy()

    windows["future_malicious"] = (
        windows["next_malicious_flow_count"] > 0
    ).astype(int)

    windows.drop(
        columns=["next_malicious_flow_count"],
        inplace=True,
    )

    # Metadata
    windows["capture"] = capture_name
    windows["window_start"] = (
        windows["window_id"] * WINDOW_SIZE_SECONDS
    )

    return windows


# ============================================================
# Main processing
# ============================================================

def main():

    print("=" * 70)
    print("CTU-IDSEVAL-6 TEMPORAL DATASET PREPARATION")
    print("=" * 70)

    log_files = find_log_files()

    print(f"\nFiles found: {len(log_files)}")

    all_windows = []

    for file_path in log_files:

        capture_name = file_path.stem

        print(f"\nProcessing: {capture_name}")

        df = load_conn_log(file_path)

        print(f"Flows loaded: {len(df):,}")

        df = clean_flow_data(df)

        windows = create_temporal_windows(
            df,
            capture_name,
        )

        print(
            f"Usable temporal windows: "
            f"{len(windows):,}"
        )

        all_windows.append(windows)

    # Combine captures
    temporal_df = pd.concat(
        all_windows,
        ignore_index=True,
    )

    # Ensure consistent column ordering
    columns = [
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
        "malicious_flow_count",
        "malicious_flow_ratio",
        "future_malicious",
        "capture",
        "window_id",
        "window_start",
    ]

    temporal_df = temporal_df[columns]

    # Sort by capture and temporal order
    temporal_df = temporal_df.sort_values(
        ["capture", "window_id"]
    ).reset_index(drop=True)

    # Save
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("TEMPORAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal usable windows: "
        f"{len(temporal_df):,}"
    )

    print(
        f"Total columns: "
        f"{len(temporal_df.columns)}"
    )

    print("\nWindows per capture:")

    print(
        temporal_df.groupby("capture").size()
    )

    print("\nFuture malicious target:")

    print(
        temporal_df["future_malicious"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget percentages:")

    print(
        temporal_df["future_malicious"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(
        f"\nOutput saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()