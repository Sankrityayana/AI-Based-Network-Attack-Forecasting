# AI-Based Network Attack Forecasting from Network Traffic Data

## 1. Abstract

Cybersecurity systems commonly analyze network traffic to identify malicious or suspicious activity. However, network behaviour is temporal, and malicious activity may develop through changes in traffic patterns over time.

This project presents a university-level prototype for forecasting malicious network activity from historical network traffic behaviour. The system uses the CTU-IDSEVAL-6 dataset containing Zeek-processed network-flow data. Flow records are transformed into 60-second temporal network states using aggregated traffic features. Five consecutive network states are then used as a temporal sequence for an LSTM model to forecast whether malicious activity is likely to occur in the following 60-second window.

A Logistic Regression model is implemented as a baseline for comparison. The models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, false-positive rate, and confusion matrices. An interactive Streamlit application provides a demonstration interface for inspecting network behaviour and generating forecasts.

On the current held-out test split, the Logistic Regression baseline achieved 99.27% accuracy, while the LSTM achieved 100% accuracy. These results demonstrate the feasibility of the implemented temporal forecasting prototype. However, the results are interpreted within the limitations of the dataset, including capture-specific traffic composition and persistent traffic behaviour.

---

# 2. Introduction

## 2.1 Background

Modern computer networks generate large volumes of traffic consisting of connections between hosts, services, ports, and protocols. Network security systems analyze this traffic to identify suspicious behaviour and possible attacks.

Traditional intrusion detection approaches generally focus on identifying malicious behaviour in observed traffic. Machine-learning methods can assist with this task by learning relationships between network-traffic features and known labels.

However, network traffic is inherently temporal. The behaviour of a network during one period can be related to its behaviour during previous periods. Therefore, a sequence-based approach can be used to represent how network behaviour changes over time.

This project explores the use of an LSTM neural network for forecasting malicious activity based on recent network behaviour.

## 2.2 Project Idea

Instead of treating each network-flow record independently, the project aggregates network flows into 60-second time windows.

Each time window is represented as a network state containing aggregated traffic features.

Five consecutive network states are then provided to an LSTM model.

The model predicts the probability that malicious activity will occur in the following 60-second window.

The central idea is:

```text
Historical Network Behaviour
             ↓
      Temporal States
             ↓
     Sequence of States
             ↓
            LSTM
             ↓
Future Malicious Activity
```

---

# 3. Problem Statement

The project addresses the problem of forecasting malicious activity from network-traffic behaviour.

The objective is to develop a machine-learning prototype that can learn temporal patterns from network traffic and estimate whether malicious activity is likely to occur in the next time window.

The project focuses on the following question:

> Can recent network behaviour be used to forecast malicious activity in the following network-traffic window?

---

# 4. Objectives

The main objectives of the project are:

1. Obtain and inspect network-flow data from the CTU-IDSEVAL-6 dataset.
2. Preprocess the Zeek network-flow records.
3. Convert flow-level traffic into temporal network states.
4. Represent each state using aggregated network-traffic features.
5. Construct sequences of consecutive network states.
6. Implement Logistic Regression as a baseline model.
7. Implement an LSTM model for temporal forecasting.
8. Evaluate both models using standard classification metrics.
9. Compare the baseline and LSTM performance.
10. Develop an interactive Streamlit demonstration.
11. Document the methodology, results, limitations, and future scope.

---

# 5. Existing Approach

Traditional network intrusion detection systems generally examine observed network traffic and attempt to determine whether the traffic is benign or malicious.

A simplified representation is:

```text
Network Traffic
       ↓
Feature Extraction
       ↓
Classification
       ↓
Benign / Malicious
```

Such an approach can be useful for detecting suspicious activity, but it does not explicitly model the temporal progression of network behaviour.

This project therefore investigates a forecasting-oriented approach in which previous network states are used to estimate malicious activity in a future time window.

---

# 6. Proposed System

The proposed system represents network traffic as a sequence of temporal network states.

The overall system is:

```text
CTU-IDSEVAL-6
      ↓
Zeek Flow Logs
      ↓
Data Parsing and Preprocessing
      ↓
60-Second Time Windows
      ↓
Network-State Feature Aggregation
      ↓
Five Consecutive Network States
      ↓
   ┌───────────────┐
   │               │
   ▼               ▼
Logistic         LSTM
Regression       Forecasting
   │               │
   └───────┬───────┘
           ▼
     Model Evaluation
           ↓
Malicious Probability
           ↓
 Streamlit Dashboard
```

---

# 7. Dataset

## 7.1 CTU-IDSEVAL-6

The project uses the CTU-IDSEVAL-6 network dataset.

The dataset contains network captures processed using Zeek to generate labeled network-flow records.

The dataset includes benign and malicious traffic as well as background traffic.

The available connection logs used in this project include:

* Benign user traffic
* Malware traffic
* Port-scan traffic

The project uses the Zeek connection logs rather than directly processing PCAP files.

## 7.2 Flow-Level Information

The connection records contain information such as:

* Timestamp
* Source address
* Source port
* Destination address
* Destination port
* Protocol
* Service
* Connection duration
* Original bytes
* Response bytes
* Connection state
* Packet counts
* IP-byte counts
* Labels

The preprocessing pipeline handles differences in the available flow-log schemas.

---

# 8. Data Preprocessing

The raw Zeek logs contain missing-value representations such as `-` and empty values.

The preprocessing process performs the following operations:

1. Load the connection logs.
2. Detect the available column structure.
3. Convert numerical fields into numeric values.
4. Replace missing-value markers with appropriate missing values.
5. Fill numerical missing values with zero.
6. Fill categorical missing values with an `unknown` category.
7. Sort the records chronologically.
8. Process each network capture separately.

Separating captures prevents temporal sequences from crossing between unrelated network captures.

---

# 9. Temporal Network-State Construction

A central part of the project is converting individual network flows into temporal network states.

## 9.1 60-Second Windows

Network flows are grouped into 60-second windows.

For example:

```text
Window 1 → 00:00–00:59
Window 2 → 01:00–01:59
Window 3 → 02:00–02:59
Window 4 → 03:00–03:59
```

Each window represents the network behaviour during one minute.

## 9.2 Network-State Features

The following aggregated features are used:

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

These features form the representation of the network state.

---

# 10. Forecasting Target

The forecasting target is defined using the following network window.

For a current network state at time `t`, the model attempts to forecast malicious activity in the following state:

```text
State(t) → State(t+1)
```

The target is binary:

```text
0 → No malicious activity in next window
1 → Malicious activity in next window
```

Therefore, the project is not predicting individual packets. It forecasts malicious activity at the 60-second network-window level.

---

# 11. Temporal Sequence Construction

The LSTM receives five consecutive network states.

For example:

```text
State(t-4)
State(t-3)
State(t-2)
State(t-1)
State(t)
        ↓
       LSTM
        ↓
Forecast State(t+1)
```

Each state contains 13 model features.

Therefore, each LSTM input has the shape:

```text
5 × 13
```

The five states represent approximately five minutes of historical network behaviour.

Sequences are constructed separately within each network capture.

---

# 12. Dataset Preparation Results

The temporal preparation produced:

* **14,711 usable 60-second network windows**
* **10,280 training sequences**
* **2,190 validation sequences**
* **2,195 test sequences**
* **5 time steps per sequence**
* **13 model input features**

The target distribution across the temporal dataset was:

| Target                       | Count |
| ---------------------------- | ----: |
| Benign next-window target    | 6,143 |
| Malicious next-window target | 8,568 |

The data is split chronologically within each capture:

* 70% training
* 15% validation
* 15% testing

---

# 13. Baseline Model

## 13.1 Logistic Regression

Logistic Regression is used as the baseline machine-learning model.

The five temporal network states are flattened into a single feature vector.

The baseline uses:

* StandardScaler
* Logistic Regression
* Maximum iterations: 2,000
* Balanced class weights
* Random state: 42

The purpose of the baseline is to establish a simple reference point against which the temporal LSTM can be compared.

---

# 14. LSTM Forecasting Model

## 14.1 Why LSTM?

LSTM networks are designed to work with sequential data.

In this project, the order of the network states is important because the model receives historical traffic behaviour as a sequence.

The LSTM therefore provides a natural model for the temporal forecasting problem.

## 14.2 Architecture

The implemented architecture is:

```text
Input: 5 × 13
       ↓
LSTM: 64 units
       ↓
Dropout: 20%
       ↓
Dense: 32 units, ReLU
       ↓
Dense: 1 unit, Sigmoid
```

The sigmoid output represents the predicted probability of malicious activity in the next network window.

## 14.3 Training

The LSTM uses:

* Adam optimizer
* Binary cross-entropy loss
* Batch size: 64
* Maximum epochs: 30
* Early stopping
* Model checkpointing
* Random seed: 42

Feature scaling is fitted using training data and then applied to validation and test data.

---

# 15. Experimental Setup

The experiment compares two approaches:

### Model 1

Logistic Regression baseline.

### Model 2

LSTM temporal forecasting model.

Both models are evaluated using the same held-out test sequences.

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* False Positive Rate
* Confusion Matrix

---

# 16. Results

## 16.1 Model Comparison

The final test-set results are:

| Model               |    Accuracy |   Precision |   Recall |          F1 |  ROC-AUC |    FPR |
| ------------------- | ----------: | ----------: | -------: | ----------: | -------: | -----: |
| Logistic Regression |      99.27% |      98.53% |     100% |      99.26% |   99.12% |  1.42% |
| **LSTM**            | **100.00%** | **100.00%** | **100%** | **100.00%** | **100%** | **0%** |

## 16.2 Logistic Regression Confusion Matrix

```text
[[1108   16]
 [   0 1071]]
```

This corresponds to:

* True negatives: 1108
* False positives: 16
* False negatives: 0
* True positives: 1071

## 16.3 LSTM Confusion Matrix

```text
[[1124    0]
 [   0 1071]]
```

This corresponds to:

* True negatives: 1124
* False positives: 0
* False negatives: 0
* True positives: 1071

---

# 17. Training Results

The LSTM training history was recorded for 30 epochs.

Final training and validation values were:

* Training accuracy: 99.9805%
* Validation accuracy: 100%
* Training loss: approximately 0.001532
* Validation loss: approximately 0.00000009

Training curves were generated for:

* Training vs validation loss
* Training vs validation accuracy

These visualizations are stored in the `results/` directory.

---

# 18. Sanity Checks

A separate sanity-check process was performed on the temporal dataset.

The direct target-leakage check passed.

The target column was not included directly as a model input feature.

Feature relationships with the forecasting target were also inspected.

The strongest observed relationship was associated with successful connection count, followed by mean duration and rejected connection count.

These checks were performed to reduce the risk of incorrectly constructing the forecasting dataset.

---

# 19. Streamlit Application

An interactive Streamlit application was developed to demonstrate the trained LSTM model.

The dashboard provides:

* Network-capture selection
* Forecast-window selection
* Recent network-state information
* Traffic statistics
* Predicted malicious probability
* Forecast classification
* Actual next-window comparison
* LSTM test metrics
* Traffic trend visualization

The application provides a practical demonstration layer on top of the trained forecasting model.

## Running the Application

```powershell
python -m streamlit run app/streamlit_app.py
```

The application can then be accessed locally through:

```text
http://localhost:8501
```

---

# 20. Project Architecture

The final architecture can be summarized as:

```text
              CTU-IDSEVAL-6
                     │
                     ▼
              Zeek Flow Logs
                     │
                     ▼
          Parsing & Preprocessing
                     │
                     ▼
            60-Second Windows
                     │
                     ▼
          Network-State Features
                     │
                     ▼
          5-State Temporal Sequence
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
      Logistic Regression   LSTM
           Baseline       Forecasting
              │             │
              └──────┬──────┘
                     ▼
              Model Evaluation
                     │
                     ▼
        Malicious Activity Forecast
                     │
                     ▼
              Streamlit UI
```

---

# 21. Advantages of the Proposed Approach

The prototype provides several useful characteristics:

1. It represents network behaviour temporally rather than treating every flow independently.
2. It uses an LSTM specifically for sequence modelling.
3. It includes a simpler baseline for comparison.
4. It provides quantitative evaluation using multiple metrics.
5. It includes visualization of model performance.
6. It provides an interactive demonstration through Streamlit.
7. The processing and modelling pipeline is reproducible through Python scripts.

---

# 22. Limitations

The current prototype has several limitations.

## 22.1 Dataset Limitation

The evaluation is based on CTU-IDSEVAL-6 rather than live enterprise network traffic.

## 22.2 Capture Composition

The different captures have different traffic compositions, and some contain persistent benign or malicious behaviour.

This can make the forecasting task easier than a real-world environment where network behaviour changes continuously.

## 22.3 Temporal Resolution

The system operates at a 60-second window level.

It does not currently make predictions for individual packets or individual flows.

## 22.4 Attack Specificity

The current system forecasts malicious activity but does not identify the exact attack technique.

## 22.5 Deployment

The prototype is designed for educational and research demonstration purposes rather than production deployment.

## 22.6 Evaluation Generalization

The current results are based on the selected held-out test split and should not be interpreted as universal performance on unseen real-world environments.

---

# 23. Interpretation of Results

The LSTM achieved perfect metrics on the current held-out test split, while Logistic Regression also achieved very high performance.

The LSTM produced:

* 100% accuracy
* 100% precision
* 100% recall
* 100% F1-score
* 100% ROC-AUC
* 0% false-positive rate

However, these results must be interpreted carefully.

The dataset contains capture-specific and persistent traffic characteristics. Consequently, the test split may not represent the difficulty of a completely new real-world network environment.

Therefore, the experiment demonstrates that the implemented pipeline can effectively model the selected temporal network data, but it does not establish that the system would achieve the same performance on arbitrary real-world traffic.

---

# 24. Future Scope

The project can be extended in several directions.

## 24.1 Whole-Dataset Forecasting

Instead of manually selecting an individual forecasting window, the application could process an entire network capture and generate forecasts for all eligible windows.

## 24.2 Real-Time Network Monitoring

The model could eventually receive continuously generated network-flow information and provide rolling forecasts.

## 24.3 PCAP Processing

A future version could directly process packet captures and generate the required flow features automatically.

## 24.4 Attack-Stage Forecasting

The system could be extended to forecast stages of an attack rather than only binary malicious activity.

## 24.5 MITRE ATT&CK Mapping

Future versions could map predicted behaviour to relevant MITRE ATT&CK techniques.

## 24.6 Explainable AI

Methods such as SHAP or attention-based analysis could be investigated to explain which network features contribute to a prediction.

## 24.7 Graph-Based Modelling

Network relationships between hosts could be represented as graphs and modelled using graph neural networks.

## 24.8 Additional Datasets

Testing on multiple independent network-security datasets would provide a stronger evaluation of generalization.

---

# 25. Project Scope

This project intentionally remains a university-level prototype.

The following are outside the current scope:

* Production-grade IDS
* SIEM implementation
* Automated incident response
* Automated attack blocking
* Live enterprise network monitoring
* Direct PCAP processing
* MITRE ATT&CK technique prediction
* Graph neural networks
* Production cloud deployment

The purpose is to demonstrate the concept of temporal network-attack forecasting using AI/ML.

---

# 26. Conclusion

This project developed an AI-based prototype for forecasting malicious network activity from historical network traffic.

The system converts Zeek network flows into 60-second network states, constructs sequences of five consecutive states, and uses an LSTM to forecast malicious activity in the following network window.

A Logistic Regression baseline was implemented to provide a reference for comparison.

The final experiment produced very high performance on the current held-out test split, with the LSTM achieving 100% accuracy and zero false positives and false negatives.

The project also includes an interactive Streamlit dashboard that demonstrates the forecasting process and presents network-traffic information and model predictions.

Although the current results are promising for the selected dataset, the dataset's capture-specific characteristics and persistent traffic behaviour limit the extent to which the results can be generalized to real-world networks.

Overall, the project demonstrates a complete university-level pipeline combining cybersecurity data, temporal feature engineering, machine learning, sequence modelling, evaluation, visualization, and interactive application development.

---

# 27. Project Deliverables

The completed project contains:

* CTU-IDSEVAL-6 dataset processing pipeline
* Network-flow preprocessing
* Temporal network-state construction
* Sequence-generation pipeline
* Logistic Regression baseline
* LSTM forecasting model
* Model evaluation scripts
* Model comparison
* Sanity-check analysis
* Result visualizations
* LSTM training-history visualizations
* Streamlit forecasting dashboard
* Architecture documentation
* Methodology and results documentation
* Presentation content
* Two-minute demonstration script
* Git version control

---

# 28. Repository Structure

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
│   ├── final_report.md
│   ├── presentation.md
│   ├── demo_script.md
│   └── project_notes.md
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
│   ├── validate_temporal_dataset.py
│   ├── sequence_builder.py
│   ├── prepare_lstm_data.py
│   ├── baseline.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── sanity_check.py
│   ├── inspect_ctu_dataset.py
│   ├── plot_results.py
│   ├── plot_training_history.py
│   └── test_environment.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 29. Final Statement

The project demonstrates how temporal modelling can be applied to network-security data to move from simple traffic classification toward forecasting-oriented analysis.

The implemented prototype provides a complete workflow from network-flow data preprocessing to temporal sequence modelling, model evaluation, and interactive forecasting visualization.
