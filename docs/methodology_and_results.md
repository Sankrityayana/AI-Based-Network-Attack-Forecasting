# Methodology and Experimental Results

## 1. Methodology

The project follows a sequential machine-learning pipeline:

1. Load CTU-IDSEVAL-6 Zeek connection logs.
2. Parse the flow records and normalize missing values.
3. Sort flows chronologically within each network capture.
4. Divide each capture into 60-second time windows.
5. Aggregate flow-level information into network-state features.
6. Define the forecasting target from malicious activity in the following window.
7. Construct sequences containing five consecutive network states.
8. Split sequences chronologically into training, validation, and test sets.
9. Train Logistic Regression as the baseline.
10. Scale the temporal features using training data and train the LSTM forecasting model.
11. Evaluate both models on the held-out test set.
12. Visualize the results and expose the LSTM through a Streamlit application.

## 2. Dataset Processing

The temporal preparation produced:

- 14,711 usable 60-second network windows
- 10,280 training sequences
- 2,190 validation sequences
- 2,195 test sequences
- 5 time steps per sequence
- 13 model input features

The target distribution across the temporal dataset was:

- Benign next-window target: 6,143
- Malicious next-window target: 8,568

The temporal sequence construction was performed separately for each capture so that sequences did not cross capture boundaries.

## 3. Baseline

The baseline model is Logistic Regression.

Each five-window sequence is flattened into a single feature vector before classification.

The baseline uses:

- StandardScaler
- LogisticRegression
- Maximum iterations: 2,000
- Balanced class weights
- Random state: 42

## 4. LSTM Model

The main model uses the following architecture:

Input: 5 time steps × 13 features
    ↓
LSTM: 64 units
    ↓
Dropout: 20%
    ↓
Dense: 32 units, ReLU
    ↓
Dense: 1 unit, Sigmoid

The model is trained using:

- Adam optimizer
- Binary cross-entropy loss
- Batch size: 64
- Maximum epochs: 30
- Early stopping based on validation loss
- Model checkpointing
- Random seed: 42

## 5. Final Test Results

### Logistic Regression

- Accuracy: 99.27%
- Precision: 98.53%
- Recall: 100%
- F1-score: 99.26%
- ROC-AUC: 99.12%
- False Positive Rate: 1.42%

Confusion matrix:

[[1108, 16],
 [0, 1071]]

### LSTM

- Accuracy: 100.00%
- Precision: 100.00%
- Recall: 100%
- F1-score: 100%
- ROC-AUC: 100%
- False Positive Rate: 0%

Confusion matrix:

[[1124, 0],
 [0, 1071]]

## 6. Model Comparison

The LSTM achieved higher test-set metrics than the Logistic Regression baseline on the current experimental split.

The main comparison is:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 99.27% | 98.53% | 100% | 99.26% | 99.12% | 1.42% |
| LSTM | 100.00% | 100% | 100% | 100% | 100% | 0% |

The LSTM produced zero false positives and zero false negatives on the current held-out test set.

## 7. Training Behaviour

The LSTM was trained for 30 epochs.

The generated training history shows:

- Final training accuracy: 99.9805%
- Final validation accuracy: 100%
- Final training loss: approximately 0.001532
- Final validation loss: approximately 0.00000009

Training-loss and training-accuracy plots are stored in the results directory.

## 8. Sanity Checks

A separate sanity-check process was used to examine the prepared temporal dataset.

The direct target-leakage check passed: the target column was not included directly among the model input features.

Feature-target relationships were also inspected.

The strongest observed correlation with the future-malicious target was associated with successful connection count, followed by mean duration and rejected connection count.

## 9. Important Interpretation

The very high test performance should be interpreted within the context of the dataset and experimental design.

The dataset contains separate captures with substantially different traffic compositions. Some captures contain persistent benign or malicious behaviour, which can make the forecasting task easier than a deployment scenario involving continuously changing real-world traffic.

Therefore, the current results demonstrate that the implemented pipeline performs very well on the selected held-out sequences, but they should not be interpreted as evidence of universal 100% real-world attack forecasting accuracy.

## 10. Limitations

The current prototype has several limitations:

- Evaluation is based on the CTU-IDSEVAL-6 dataset rather than live enterprise traffic.
- The system operates on aggregated 60-second windows.
- Predictions are made for the next window rather than individual packets.
- The current model does not identify specific attack techniques.
- The current evaluation is dataset-specific.
- Capture composition may contribute to the unusually high test performance.
- The prototype is not intended to replace an operational IDS or SIEM.

## 11. Reproducibility

The project contains scripts for:

- Dataset inspection
- Dataset preparation
- Temporal validation
- Sequence generation
- Baseline training
- LSTM training
- Evaluation
- Model comparison
- Result visualization
- Training-history visualization
- Streamlit demonstration

The generated evaluation files and plots are stored under the esults/ directory.

## 12. Final Experimental Conclusion

The experiment demonstrates a complete temporal network-traffic forecasting pipeline.

The Logistic Regression baseline provides a strong non-temporal reference, while the LSTM models the sequence of recent network states.

On the current held-out test split, the LSTM achieved perfect classification metrics and a lower false-positive rate than the Logistic Regression baseline.

These results support the feasibility of the proposed university-level prototype while also highlighting the need for broader datasets and more challenging evaluation scenarios before making claims about real-world deployment.
