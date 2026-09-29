# AI-Based Network Attack Forecasting from Network Traffic Data

## Slide 1 — Problem & Objective

### Problem

* Traditional intrusion detection primarily identifies suspicious activity in observed traffic.
* Network attacks can involve changing behaviour over time.
* The objective is to use recent network behaviour to forecast malicious activity in the next time window.

### Objective

> Build an AI/ML prototype that learns temporal network-traffic patterns and predicts the probability of malicious activity in the next 60-second window.

### Key Idea

```text
Past network behaviour
          ↓
Temporal model
          ↓
Future malicious-activity forecast
```

---

# Slide 2 — Dataset & Proposed Approach

## Dataset: CTU-IDSEVAL-6

* Zeek-processed network-flow data
* Contains benign, malicious and background traffic
* Multiple network captures
* Flow-level features such as:

  * Protocol
  * Duration
  * Bytes
  * Packets
  * Connection state
  * Ports

## Our Approach

```text
Raw Zeek Flows
      ↓
Preprocessing
      ↓
60-second Network Windows
      ↓
13 Network-State Features
      ↓
5 Consecutive Windows
      ↓
LSTM Forecast
```

### Target

Predict malicious activity in the following **60-second network window**.

---

# Slide 3 — System Architecture & Methodology

## System Architecture

```text
       CTU-IDSEVAL-6
              ↓
       Zeek Flow Logs
              ↓
    Data Preprocessing
              ↓
     60-sec Time Windows
              ↓
   Network State Features
              ↓
      5-Window Sequence
              ↓
       ┌──────┴──────┐
       ↓             ↓
 Logistic Regression  LSTM
    Baseline        Forecasting
       │             │
       └──────┬──────┘
              ↓
        Model Evaluation
              ↓
     Malicious Probability
              ↓
     Streamlit Dashboard
```

## LSTM Architecture

```text
Input (5 × 13)
       ↓
LSTM (64)
       ↓
Dropout (20%)
       ↓
Dense (32, ReLU)
       ↓
Sigmoid
       ↓
Malicious Probability
```

---

# Slide 4 — Models & Results

## Test-Set Comparison

| Model               | Accuracy | Precision |   Recall |       F1 |  ROC-AUC |
| ------------------- | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression |   99.27% |    98.53% |     100% |   99.26% |   99.12% |
| **LSTM**            | **100%** |  **100%** | **100%** | **100%** | **100%** |

## LSTM Test Confusion Matrix

```text
                  Predicted
                  Benign  Malicious

Actual Benign       1124      0
Actual Malicious       0    1071
```

## Experimental Observation

The LSTM achieved perfect results on the current held-out test split.

However, the dataset contains capture-specific and persistent traffic patterns. Therefore, these results should **not** be interpreted as 100% real-world forecasting accuracy.

---

# Slide 5 — Demo, Limitations & Future Work

## Prototype Demo

The Streamlit application provides:

* Network-capture selection
* Forecast-window selection
* Recent network-state information
* Traffic statistics
* Malicious probability
* Forecast classification
* Actual next-window comparison
* Model performance metrics

## Current Limitations

* Dataset-based rather than live monitoring
* Uses 60-second aggregated network states
* Forecasts malicious activity rather than individual packets
* No SIEM functionality
* No automated incident response
* No MITRE ATT&CK mapping
* No production deployment claims

## Future Work

* Whole-dataset forecasting analysis
* Real-time traffic ingestion
* PCAP processing
* Attack-stage prediction
* MITRE ATT&CK mapping
* Explainable AI
* Graph-based network modelling
* Evaluation on additional datasets

---

# Presentation Story

The presentation follows one simple flow:

```text
Problem
   ↓
Dataset
   ↓
Temporal Network Representation
   ↓
LSTM Forecasting
   ↓
Results
   ↓
Streamlit Demonstration
```

### Core Message

> **The project transforms historical network traffic into temporal network states and uses an LSTM to forecast the likelihood of malicious activity in the next 60-second window.**
