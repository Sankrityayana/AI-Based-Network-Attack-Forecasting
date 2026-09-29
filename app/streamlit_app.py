from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_FILE = (
    MODEL_DIR /
    "lstm_attack_forecasting.keras"
)

SCALER_FILE = (
    MODEL_DIR /
    "lstm_feature_scaler.joblib"
)

TEMPORAL_DATA_FILE = (
    DATA_DIR /
    "temporal_network_states.csv"
)

FEATURES = [
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

SEQUENCE_LENGTH = 5


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Network Attack Forecasting",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .prediction-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .small-text {
        color: #666;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_FILE
    )


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    return joblib.load(
        SCALER_FILE
    )


# ============================================================
# LOAD TEMPORAL DATASET
# ============================================================

@st.cache_data
def load_temporal_data():

    df = pd.read_csv(
        TEMPORAL_DATA_FILE
    )

    return df


# ============================================================
# LOAD MODEL METRICS
# ============================================================

@st.cache_data
def load_metrics():

    metrics = {}

    metrics_file = (
        RESULTS_DIR /
        "lstm_metrics.txt"
    )

    if not metrics_file.exists():
        return metrics

    with open(
        metrics_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if ":" not in line:
                continue

            key, value = line.split(
                ":",
                1
            )

            key = key.strip()
            value = value.strip()

            try:
                metrics[key] = float(value)
            except ValueError:
                pass

    return metrics


# ============================================================
# FORMAT LARGE NUMBERS
# ============================================================

def format_number(value):

    if pd.isna(value):
        return "N/A"

    value = float(value)

    if value >= 1_000_000:

        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:

        return f"{value / 1_000:.2f}K"

    return f"{value:.2f}"


# ============================================================
# CREATE SEQUENCE
# ============================================================

def create_sequence(
    capture_df,
    selected_index
):

    # Need five previous windows.
    if selected_index < SEQUENCE_LENGTH:
        return None

    sequence_df = capture_df.iloc[
        selected_index - SEQUENCE_LENGTH:
        selected_index
    ]

    values = sequence_df[
        FEATURES
    ].values.astype(
        np.float32
    )

    return values


# ============================================================
# SCALE SEQUENCE
# ============================================================

def scale_sequence(
    sequence,
    scaler
):

    original_shape = sequence.shape

    flattened = sequence.reshape(
        -1,
        len(FEATURES)
    )

    scaled = scaler.transform(
        flattened
    )

    return scaled.reshape(
        original_shape
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_next_window(
    model,
    scaler,
    sequence
):

    scaled_sequence = scale_sequence(
        sequence,
        scaler
    )

    model_input = np.expand_dims(
        scaled_sequence,
        axis=0
    )

    probability = float(
        model.predict(
            model_input,
            verbose=0
        )[0][0]
    )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    return probability, prediction


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ AI-Based Network Attack Forecasting'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'LSTM-based temporal forecasting of malicious '
    'network activity from Zeek network-flow data'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    model = load_model()
    scaler = load_scaler()
    df = load_temporal_data()
    metrics = load_metrics()

except Exception as error:

    st.error(
        "Unable to load the project resources."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Forecast Controls"
)

st.sidebar.markdown(
    """
    The LSTM uses the previous **5 one-minute
    network states** to forecast malicious activity
    in the next one-minute window.
    """
)

captures = sorted(
    df["capture"].unique()
)

selected_capture = st.sidebar.selectbox(
    "Select network capture",
    captures
)


# ============================================================
# FILTER CAPTURE
# ============================================================

capture_df = (
    df[
        df["capture"] ==
        selected_capture
    ]
    .sort_values(
        "window_id"
    )
    .reset_index(
        drop=True
    )
)


if len(capture_df) <= SEQUENCE_LENGTH:

    st.warning(
        "This capture does not contain enough "
        "temporal windows to create a 5-window "
        "forecast sequence."
    )

    st.stop()


# Only windows for which five previous windows exist.
valid_indices = list(
    range(
        SEQUENCE_LENGTH,
        len(capture_df)
    )
)


selected_index = st.sidebar.slider(
    "Forecast window",
    min_value=SEQUENCE_LENGTH,
    max_value=len(capture_df) - 1,
    value=SEQUENCE_LENGTH,
    step=1,
)


# ============================================================
# CURRENT WINDOW INFORMATION
# ============================================================

current_row = capture_df.iloc[
    selected_index
]


window_start = float(
    current_row["window_start"]
)


window_id = int(
    current_row["window_id"]
)


# ============================================================
# PREDICTION
# ============================================================

sequence = create_sequence(
    capture_df,
    selected_index
)

probability, prediction = predict_next_window(
    model,
    scaler,
    sequence
)


# ============================================================
# MAIN PREDICTION
# ============================================================

st.subheader(
    "Forecast Result"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Malicious Probability",
        f"{probability * 100:.2f}%"
    )


with col2:

    if prediction == 1:

        st.metric(
            "Forecast",
            "MALICIOUS"
        )

    else:

        st.metric(
            "Forecast",
            "BENIGN"
        )


with col3:

    st.metric(
        "Forecast Horizon",
        "Next 60 seconds"
    )


# ============================================================
# INTERPRETATION
# ============================================================

if prediction == 1:

    st.error(
        "⚠️ The LSTM forecasts malicious activity "
        "in the next network-traffic window."
    )

else:

    st.success(
        "✓ The LSTM forecasts no malicious activity "
        "in the next network-traffic window."
    )


st.caption(
    "Prediction is based on the previous five "
    "one-minute network-traffic windows."
)


# ============================================================
# CURRENT NETWORK STATISTICS
# ============================================================

st.subheader(
    "Current Network State"
)

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:

    st.metric(
        "Flows",
        format_number(
            current_row["flow_count"]
        )
    )

with stat2:

    st.metric(
        "Original Bytes",
        format_number(
            current_row["total_orig_bytes"]
        )
    )

with stat3:

    st.metric(
        "Response Bytes",
        format_number(
            current_row["total_resp_bytes"]
        )
    )

with stat4:

    st.metric(
        "Mean Duration",
        f"{current_row['mean_duration']:.2f}s"
    )


# ============================================================
# RECENT TEMPORAL WINDOW
# ============================================================

st.subheader(
    "Recent Network-Traffic Windows"
)

recent_df = capture_df.iloc[
    selected_index - SEQUENCE_LENGTH:
    selected_index
].copy()

recent_display = recent_df[
    [
        "window_id",
        "window_start",
        "flow_count",
        "total_orig_bytes",
        "total_resp_bytes",
        "tcp_flow_count",
        "udp_flow_count",
        "successful_connection_count",
        "rejected_connection_count",
    ]
].copy()

recent_display.columns = [
    "Window",
    "Start Time",
    "Flows",
    "Original Bytes",
    "Response Bytes",
    "TCP Flows",
    "UDP Flows",
    "Successful",
    "Rejected",
]

st.dataframe(
    recent_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TRAFFIC TREND
# ============================================================

st.subheader(
    "Recent Traffic Trend"
)

chart_df = recent_df[
    [
        "window_id",
        "flow_count",
        "tcp_flow_count",
        "udp_flow_count",
    ]
].copy()

chart_df = chart_df.set_index(
    "window_id"
)

chart_df.columns = [
    "Total Flows",
    "TCP Flows",
    "UDP Flows",
]

st.line_chart(
    chart_df
)


# ============================================================
# ACTUAL NEXT WINDOW
# ============================================================

st.subheader(
    "Actual Next-Window Label"
)

if "future_malicious" in current_row.index:

    actual = int(
        current_row["future_malicious"]
    )

    actual_label = (
        "MALICIOUS"
        if actual == 1
        else "BENIGN"
    )

    actual_col1, actual_col2 = st.columns(2)

    with actual_col1:

        st.metric(
            "Actual Label",
            actual_label
        )

    with actual_col2:

        if actual == prediction:

            st.success(
                "Forecast matched actual label."
            )

        else:

            st.warning(
                "Forecast did not match actual label."
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.subheader(
    "LSTM Test Performance"
)

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

with metric1:

    st.metric(
        "Accuracy",
        f"{metrics.get('Accuracy', 0) * 100:.2f}%"
    )

with metric2:

    st.metric(
        "Precision",
        f"{metrics.get('Precision', 0) * 100:.2f}%"
    )

with metric3:

    st.metric(
        "Recall",
        f"{metrics.get('Recall', 0) * 100:.2f}%"
    )

with metric4:

    st.metric(
        "F1",
        f"{metrics.get('F1', 0) * 100:.2f}%"
    )

with metric5:

    st.metric(
        "ROC-AUC",
        f"{metrics.get('ROC-AUC', 0) * 100:.2f}%"
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander(
    "About this prototype"
):

    st.markdown(
        """
        ### Project

        **AI Based Network Attack Forecasting from
        Network Traffic Data**

        ### Dataset

        CTU-IDSEVAL-6 Zeek flow data.

        ### Temporal representation

        Network traffic is aggregated into
        **60-second windows**.

        ### Forecasting approach

        The LSTM receives the previous **5 network
        states** and predicts whether malicious
        activity will occur in the next window.

        ### Model

        - LSTM: 64 units
        - Dropout: 20%
        - Dense layer: 32 units
        - Sigmoid output
        - Binary crossentropy
        - Adam optimizer

        ### Prototype scope

        This is a university-level research prototype.
        It is not intended to replace a production IDS
        or provide real-world security response.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "University-level prototype • "
    "Temporal network-state forecasting using LSTM"
)