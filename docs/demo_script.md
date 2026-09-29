# 2-Minute Project Demo

## 0:00–0:20 — Problem

> Our project focuses on forecasting malicious network activity from network traffic. Instead of only asking whether currently observed traffic is malicious, we use recent network behaviour to estimate whether malicious activity is likely in the next time window.

---

## 0:20–0:40 — Data & Approach

> We use the CTU-IDSEVAL-6 Zeek network-flow dataset. We aggregate flows into 60-second network states and use the previous five states as a temporal sequence.

### Pipeline

```text
CTU-IDSEVAL-6
      ↓
Zeek Network Flows
      ↓
60-Second Network States
      ↓
Previous 5 States
      ↓
LSTM Forecast
```

---

## 0:40–1:10 — Model

> We implemented two models: Logistic Regression as a baseline and an LSTM as our temporal forecasting model. The LSTM receives five consecutive network states and predicts the probability of malicious activity in the following 60-second window.

### LSTM

```text
Input: 5 × 13
      ↓
LSTM (64)
      ↓
Dropout (20%)
      ↓
Dense (32)
      ↓
Sigmoid
      ↓
Malicious Probability
```

---

## 1:10–1:40 — Live Dashboard

Open the Streamlit application and demonstrate:

1. Select a network capture.
2. Select a forecasting window.
3. Show the previous five network states.
4. Show the predicted malicious probability.
5. Show the forecast classification.
6. Show the traffic statistics and trend.

### Say:

> The dashboard allows us to inspect the recent network behaviour and see the model's forecast for the next window.

---

## 1:40–2:00 — Results

Show the model comparison:

| Model               | Accuracy | Precision |   Recall |       F1 |  ROC-AUC |
| ------------------- | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression |   99.27% |    98.53% |     100% |   99.26% |   99.12% |
| **LSTM**            | **100%** |  **100%** | **100%** | **100%** | **100%** |

### Say:

> On the current held-out test split, Logistic Regression achieved 99.27% accuracy, while the LSTM achieved 100%. We also evaluated precision, recall, F1-score, ROC-AUC and false-positive rate. Because the dataset has capture-specific and persistent traffic patterns, we treat these results as experimental results rather than claiming 100% real-world accuracy.

---

# Demo Flow

```text
Problem
   ↓
Dataset
   ↓
5 Temporal Network States
   ↓
LSTM
   ↓
Forecast
   ↓
Streamlit Dashboard
   ↓
Results
```

## Core Message

> **The project transforms historical network traffic into temporal network states and uses an LSTM to forecast the likelihood of malicious activity in the next 60-second window.**
