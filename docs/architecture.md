# AI-Based Network Attack Forecasting from Network Traffic Data

## 1. Project Overview

This project is a university-level prototype for forecasting malicious network activity from historical network traffic behaviour.

The system converts network-flow data into time-based network states and uses a sequence of previous network states to predict whether malicious activity is likely to occur in the next time window.

## 2. Problem

Traditional network intrusion detection focuses primarily on identifying malicious activity in observed traffic.

This project explores a forecasting-oriented approach: learning temporal patterns in network traffic and using recent network behaviour to estimate the likelihood of malicious activity in the next time window.

## 3. Dataset

The project uses the CTU-IDSEVAL-6 network traffic dataset.

The dataset contains Zeek-processed network flows with benign, malicious, and background traffic labels.

The project processes the available connection logs and constructs temporal network states from the flow-level data.

## 4. System Architecture

The overall pipeline is:

Dataset
    ↓
Zeek Flow Logs
    ↓
Data Parsing & Preprocessing
    ↓
60-Second Time Windows
    ↓
Network-State Feature Aggregation
    ↓
Five Consecutive Network States
    ↓
    ┌──────────────────────┐
    │                      │
    ▼                      ▼
Logistic Regression       LSTM
    │                      │
    └──────────┬───────────┘
               ▼
        Model Evaluation
               ↓
     Malicious Activity Forecast
               ↓
       Streamlit Dashboard

## 5. Temporal Network State Construction

Individual network flows are grouped into 60-second time windows.

For each window, traffic is summarized using features including:

- Flow count
- Total original bytes
- Total response bytes
- Total original packets
- Total response packets
- Mean connection duration
- Total missed bytes
- Unique source ports
- Unique destination ports
- TCP flow count
- UDP flow count
- Successful connection count
- Rejected connection count

These aggregated features represent the network state during each time window.

## 6. Sequence Construction

The forecasting model uses five consecutive network states as its input sequence.

For a sequence:

State(t-4)
State(t-3)
State(t-2)
State(t-1)
State(t)

the model predicts malicious activity in the following time window.

Therefore, the LSTM receives approximately five minutes of historical network behaviour and produces a forecast for the next 60-second window.

## 7. Baseline Model

Logistic Regression is used as a baseline machine-learning model.

The five network states are flattened into a feature vector and provided to the classifier.

This establishes a simple reference point against which the temporal LSTM model can be compared.

## 8. LSTM Forecasting Model

The main forecasting model is an LSTM neural network.

Architecture:

Input (5 time steps × 13 features)
        ↓
LSTM (64 units)
        ↓
Dropout (20%)
        ↓
Dense (32 units, ReLU)
        ↓
Dense (1 unit, Sigmoid)
        ↓
Malicious probability

The sigmoid output represents the predicted probability of malicious activity in the next network window.

## 9. Data Splitting

The temporal data is divided chronologically within each capture into:

- 70% training data
- 15% validation data
- 15% test data

Sequences are constructed without crossing between separate network captures.

Feature scaling for the LSTM is fitted using training data only.

## 10. Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- False Positive Rate
- Confusion Matrix

The baseline and LSTM models are evaluated on the same held-out test sequences.

## 11. Application Layer

A Streamlit application provides an interactive demonstration of the trained LSTM model.

The application allows the user to:

- Select a network capture
- Select a forecasting window
- View recent network states
- View network traffic statistics
- Obtain the predicted malicious probability
- View the forecast classification
- Compare the forecast with the available next-window label
- View model performance information

## 12. Project Scope

This prototype focuses specifically on temporal malicious-activity forecasting from network-flow data.

It does not attempt to provide:

- Production-grade intrusion detection
- SIEM functionality
- Automated incident response
- Live network packet capture
- Automatic attack blocking
- Direct PCAP processing
- MITRE ATT&CK technique classification
- Graph neural networks

These features are outside the scope of the university-level prototype.

## 13. Expected Outcome

The final system demonstrates how historical network behaviour can be represented as temporal network states and processed using sequence modelling to forecast malicious activity in a future time window.

The project also compares the temporal LSTM approach with a Logistic Regression baseline and provides an interactive Streamlit demonstration.
