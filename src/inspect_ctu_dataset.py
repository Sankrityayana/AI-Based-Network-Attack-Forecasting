from pathlib import Path
import pandas as pd


# ============================================================
# CTU-IDSEVAL-6 DATASET INSPECTION
# ============================================================

DATA_DIR = Path("data/raw/CTU-IDSEVAL-6/zeek/zeek")


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
]

COLUMNS_23 = BASE_COLUMNS + [
    "label",
    "detailedlabel",
]

COLUMNS_24 = BASE_COLUMNS + [
    "ip_proto",
    "label",
    "detailedlabel",
]


# ============================================================
# LOAD ZEEK FILE
# ============================================================

def load_zeek_file(filepath):

    rows = []

    with open(
        filepath,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line in file:

            # Skip Zeek metadata
            if line.startswith("#"):
                continue

            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            # Zeek uses tab separator
            parts = line.split("\t")

            rows.append(parts)

    if not rows:
        return pd.DataFrame(columns=COLUMNS_23)

    # Determine actual format from number of fields
    field_counts = {}

    for row in rows:
        field_counts[len(row)] = field_counts.get(len(row), 0) + 1

    print(f"\nField counts detected: {field_counts}")

    # Most common field count
    detected_count = max(
        field_counts,
        key=field_counts.get
    )

    if detected_count == 23:

        columns = COLUMNS_23

    elif detected_count == 24:

        columns = COLUMNS_24

    else:

        raise ValueError(
            f"Unexpected number of fields in "
            f"{filepath.name}: {detected_count}"
        )

    valid_rows = []

    for row in rows:

        if len(row) == detected_count:

            valid_rows.append(row)

        else:

            print(
                f"WARNING: Skipping row with "
                f"{len(row)} fields in {filepath.name}"
            )

    df = pd.DataFrame(
        valid_rows,
        columns=columns
    )

    return df


# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    DATA_DIR.glob("*.conn-labeled.log")
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CTU-IDSEVAL-6 DATASET INSPECTION")
print("=" * 70)

print()

print(f"Flow files found: {len(files)}")

for file in files:

    print(f" - {file.name}")


# ============================================================
# INSPECT FILES
# ============================================================

for filepath in files:

    print()
    print("=" * 70)
    print(filepath.name)
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_zeek_file(filepath)

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    print("\nColumns:")

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:2}. {column}"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print("\nLabels:")

    print(
        df["label"].value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Detailed labels
    # --------------------------------------------------------

    print("\nDetailed Labels:")

    print(
        df["detailedlabel"].value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    print(
        df.isna().sum()
    )

    # --------------------------------------------------------
    # Convert important numeric fields
    # --------------------------------------------------------

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
                errors="coerce"
            )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\nData types:")

    print(
        df.dtypes
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    print("\nTimestamp validation:")

    valid_ts = df["ts"].notna().sum()

    invalid_ts = df["ts"].isna().sum()

    print(
        f"Valid numeric timestamps: {valid_ts}"
    )

    print(
        f"Invalid timestamps: {invalid_ts}"
    )

    if valid_ts > 0:

        print(
            f"Minimum: {df['ts'].min()}"
        )

        print(
            f"Maximum: {df['ts'].max()}"
        )

    # --------------------------------------------------------
    # First rows
    # --------------------------------------------------------

    print("\nFirst 3 rows:")

    print(
        df.head(3).to_string(
            index=False
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)