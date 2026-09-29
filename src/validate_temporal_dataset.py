from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path(
    "data/processed/temporal_network_states.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)


print("=" * 70)
print("TEMPORAL NETWORK STATE VALIDATION")
print("=" * 70)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nShape:")
print(df.shape)


print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# DATA TYPES
# ============================================================

print("\nData types:")
print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isna().sum())


# ============================================================
# TARGET
# ============================================================

print("\nFuture malicious target:")
print(
    df["future_malicious"].value_counts()
)

print("\nTarget percentages:")
print(
    df["future_malicious"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# CAPTURE DISTRIBUTION
# ============================================================

print("\nWindows per capture:")
print(
    df.groupby("capture")
    .size()
)


# ============================================================
# CHECK IMPORTANT NUMERIC FEATURES
# ============================================================

numeric_features = [
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
]


print("\nNumeric feature summary:")

print(
    df[numeric_features]
    .describe()
    .T
    .to_string()
)


# ============================================================
# CHECK FOR NEGATIVE VALUES
# ============================================================

print("\nNegative values:")

for column in numeric_features:

    negative_count = (
        df[column] < 0
    ).sum()

    if negative_count > 0:

        print(
            f"{column}: {negative_count}"
        )


# ============================================================
# SAMPLE ROWS
# ============================================================

print("\nFirst 10 rows:")

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# MALICIOUS RATIO CHECK
# ============================================================

print("\nMalicious ratio range:")

print(
    "Minimum:",
    df["malicious_flow_ratio"].min()
)

print(
    "Maximum:",
    df["malicious_flow_ratio"].max()
)


# ============================================================
# LOGICAL CHECK
# ============================================================

print("\nLogical checks:")

print(
    "flow_count > 0:",
    (df["flow_count"] > 0).all()
)

print(
    "orig_bytes >= 0:",
    (df["total_orig_bytes"] >= 0).all()
)

print(
    "resp_bytes >= 0:",
    (df["total_resp_bytes"] >= 0).all()
)

print(
    "orig_pkts >= 0:",
    (df["total_orig_pkts"] >= 0).all()
)

print(
    "resp_pkts >= 0:",
    (df["total_resp_pkts"] >= 0).all()
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)