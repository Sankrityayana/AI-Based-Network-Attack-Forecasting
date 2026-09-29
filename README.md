# AI-Based Network Attack Forecasting from Network Traffic Data

A university-level cybersecurity and AI/ML prototype that learns temporal patterns in network traffic and forecasts whether malicious activity is likely to occur in the next network-traffic window.

## Project Overview

Traditional intrusion detection focuses on identifying suspicious activity in observed traffic.

This project explores a forecasting-oriented approach:

> Given the recent behaviour of a network, can an AI model forecast whether malicious activity is likely in the next time window?

The system converts Zeek network-flow data into 60-second network states, forms sequences of five consecutive states, and uses an LSTM model to forecast malicious activity in the following window.

## Problem

Network traffic is temporal: its behaviour changes over time and an attack may involve a progression of changing network activity.

Instead of treating every flow independently, this project represents network behaviour as a sequence of temporal states and learns patterns across those states.

## Solution

The pipeline is:

```text
CTU-IDSEVAL-6
      ↓
Zeek Flow Logs
      ↓
Data Preprocessing
      ↓
60-Second Network Windows
      ↓
Network-State Feature Aggregation
      ↓
Five Consecutive States
      ↓
LSTM
      ↓
Next-Window Malicious Probability
      ↓
Streamlit Dashboard
```

A Logistic Regression model is also implemented as a baseline for comparison.

## Key Idea

The LSTM receives approximately five minutes of historical network behaviour:

```text
State(t-4)
State(t-3)
State(t-2)
State(t-1)
State(t)
```

and forecasts malicious activity in:

```text
State(t+1)
```

The prediction is expressed as a probability and converted into a benign/malicious forecast.

## Dataset

The project uses the CTU-IDSEVAL-6 network traffic dataset.

The dataset contains Zeek-processed network flows containing benign, malicious, and background traffic.

Dataset documentation is available in:

```text
docs/dataset.md
```

## Network-State Features

Each 60-second window is represented using aggregated traffic features including:

* Flow count
* Total original bytes
* Total response bytes
* Total original packets
* Total response packets
* Mean duration
* Total missed bytes
* Unique source ports
* Unique destination ports
* TCP flow count
* UDP flow count
* Successful connection count
* Rejected connection count

## Models

### Logistic Regression Baseline

A Logistic Regression classifier provides a simple reference model.

The five temporal states are flattened into a feature vector before classification.

### LSTM Forecasting Model

The main model is:

```text
Input: 5 × 13
       ↓
LSTM(64)
       ↓
Dropout(0.20)
       ↓
Dense(32, ReLU)
       ↓
Dense(1, Sigmoid)
```

The output represents the predicted probability of malicious activity in the next network window.

## Dataset and Sequence Size

The temporal preparation produced:

* 14,711 usable 60-second network windows
* 10,280 training sequences
* 2,190 validation sequences
* 2,195 test sequences
* 5 time steps per sequence
* 13 input features

Sequences are constructed separately within each network capture.

## Results

Final held-out test-set results:

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |   FPR |
| ------------------- | -------: | --------: | -----: | -----: | ------: | ----: |
| Logistic Regression |   99.27% |    98.53% |   100% | 99.26% |  99.12% | 1.42% |
| LSTM                |  100.00% |      100% |   100% |   100% |    100% |    0% |

### LSTM Confusion Matrix

```text
              Predicted
              0      1
Actual 0    1124     0
Actual 1       0  1071
```

The LSTM produced zero false positives and zero false negatives on the current held-out test split.

## Important Result Interpretation

The high test performance should be interpreted within the context of the dataset and experimental setup.

The captures have different traffic compositions and some contain persistent benign or malicious behaviour. This can make the forecasting task easier than a real-world environment with continuously changing traffic.

Therefore, the results demonstrate the performance of this prototype on the selected held-out test sequences and should not be interpreted as proof of 100% real-world attack forecasting accuracy.

## Streamlit Application

The project includes an interactive Streamlit dashboard.

The application allows users to:

* Select a network capture
* Select a forecasting window
* View recent network states
* View traffic statistics
* Obtain malicious probability
* View the forecast classification
* Compare the forecast with the available next-window label
* View model performance information

Run the application with:

```powershell
python -m streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Project Structure

```text
AI Based Network Attack Forecasting/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── methodology_and_results.md
│   ├── project_notes.md
│   ├── labels_contents.txt
│   └── zeek_contents.txt
│
├── models/
│
├── results/
│   ├── baseline_metrics.txt
│   ├── lstm_metrics.txt
│   ├── model_comparison.txt
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_lstm.png
│   ├── roc_curve_comparison.png
│   ├── model_performance_comparison.png
│   ├── lstm_training_loss.png
│   └── lstm_training_accuracy.png
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── prepare_dataset.py
│   ├── sequence_builder.py
│   ├── prepare_lstm_data.py
│   ├── baseline.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── sanity_check.py
│   ├── plot_results.py
│   ├── plot_training_history.py
│   └── ...
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Main Scripts

| Script                         | Purpose                                |
| ------------------------------ | -------------------------------------- |
| `inspect_ctu_dataset.py`       | Inspect raw CTU-IDSEVAL-6 flow logs    |
| `prepare_dataset.py`           | Build temporal network states          |
| `validate_temporal_dataset.py` | Validate temporal dataset              |
| `sequence_builder.py`          | Build temporal sequences               |
| `prepare_lstm_data.py`         | Scale LSTM data                        |
| `baseline.py`                  | Train Logistic Regression              |
| `train.py`                     | Train LSTM                             |
| `evaluate.py`                  | Evaluate LSTM                          |
| `compare_models.py`            | Compare baseline and LSTM              |
| `sanity_check.py`              | Check dataset and target relationships |
| `plot_results.py`              | Generate evaluation plots              |
| `plot_training_history.py`     | Generate training curves               |

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Project

The trained models and processed results are already generated for the current experiment.

To launch the dashboard:

```powershell
python -m streamlit run app/streamlit_app.py
```

To reproduce the main modelling workflow, the scripts in `src/` can be executed in the appropriate data-processing and modelling sequence.

## Limitations

This is a university-level prototype.

It does not provide:

* Production-grade intrusion detection
* SIEM functionality
* Automated incident response
* Live enterprise network monitoring
* Automated attack blocking
* Direct PCAP processing
* Specific MITRE ATT&CK technique prediction
* Graph neural network modelling

The system operates on aggregated 60-second network states rather than individual packets.

## Project Status

**Technical prototype: Complete.**

Implemented:

* Dataset processing
* Temporal network-state construction
* Sequence modelling
* Logistic Regression baseline
* LSTM forecasting
* Evaluation
* Visualization
* Streamlit demonstration
* Git version control

## Future Extensions

Possible future research directions include:

* Whole-dataset forecasting analysis
* Real-time traffic ingestion
* PCAP-based processing
* Attack-stage prediction
* MITRE ATT&CK mapping
* Explainability methods
* Graph-based network-state modelling
* Evaluation on additional datasets

## Disclaimer

This project is an educational and research prototype developed for a university-level cybersecurity and AI/ML project. Its results should not be interpreted as production security guarantees.
