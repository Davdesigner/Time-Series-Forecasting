# Beijing Air Quality Forecasting (PM2.5 Prediction)

## Project Overview

This project focuses on forecasting PM2.5 (particulate matter ≤2.5µm) concentrations in **Beijing** using historical air quality and weather data.  
The goal is to build an accurate time-series prediction model using **Long Short-Term Memory (LSTM)** networks, enabling **proactive pollution control** measures.

---

## Key Features

-  **Data Preprocessing**: Handling missing values, normalization, and time-series structuring.  
-  **LSTM & Bidirectional LSTM Models**: Experimented with different architectures and hyperparameters.  
-  **Optimization**: Tested `Adam`, `RMSprop`, and `SGD` optimizers with learning rate tuning.  
-  **Kaggle Submission**: Generated predictions for test data and formatted for competition submission.

---

## Methodology

### 1. Data Exploration & Preprocessing

- **Missing Values**:  
  - Linear interpolation for continuity.  
  - Forward/backward filling for gaps (up to 155 consecutive hours).

- **Feature Scaling**:  
  - Used `MinMaxScaler` for normalization.

- **Visualizations**:
  - Time-series trends (seasonality, outliers).
  - Correlation heatmaps (feature importance).
  - Distribution analysis (highly skewed PM2.5 data).

---

### 2. Model Architecture

- **LSTM Variants**:
  - Standard LSTM (1–3 layers).
  - Bidirectional LSTM (achieved best performance).

- **Hyperparameters**:
  - Units: `[32, 64, 128, 256]`
  - Dropout: `0.1–0.3` (to reduce overfitting)
  - Optimizers:
    - `Adam`
    - `RMSprop` (best performance at `lr=0.0002`)
    - `SGD`
  - Activation: `tanh` / `ReLU`

---

### 3. Training & Evaluation

- **Early Stopping**:
  - Patience = 10–15 epochs to prevent overfitting.

- **Loss Metrics**:
  - `RMSE` (Root Mean Squared Error) – metric for model comparison.

- **Best Model**:
  - `Model 8` (Bidirectional LSTM) achieved the **lowest validation loss** of 5329.9 .

---

##  Results & Insights
A table containing the results and performance of each model can be found in the report file that is uploaded here. However, we have;
### Key Findings

- **Bidirectional LSTMs** outperformed standard LSTMs by capturing patterns in both directions.
- `RMSprop` with `lr=0.0002` and dropout of `0.15` yielded stable convergence.
- **Deeper LSTM networks (3+ layers)** improved performance when combined with proper regularization.
- **Missing data interpolation + forward fill** preserved time-series structure effectively.

---

### Challenges & Future Improvements

- **Outliers**: Extreme PM2.5 values skewed MSE. Future work should test **log-scaling**.
- **Feature Engineering**: Include additional features like **month**, **hour**, and **day of week**.
- **Sequence Length**: Experiment with **longer time windows (24h or more)** to improve long-term forecasting.

---