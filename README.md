# MATH8800 — Project 2: Energy Consumption Forecasting

**Author:** Ze Yu Jiang  
**Date:** May 2026

---

## Overview

Building energy consumption forecasting using the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction) dataset. It compares classical baseline methods against deep learning approaches (LSTM and Transformer) for predicting hourly meter readings across multiple building types and energy meter types.


Download the following files from the [Kaggle competition page](https://www.kaggle.com/c/ashrae-energy-prediction/data) and place them in a `data/` folder at the project root:

| File | Description |
|---|---|
| `train.csv` | Hourly meter readings per building |
| `test.csv` | Hourly meter readings for inference |
| `building_metadata.csv` | Static building attributes (size, use type, etc.) |
| `weather_train.csv` | Hourly weather observations (train period) |
| `weather_test.csv` | Hourly weather observations (test period) |

---

## Notebook: `test.ipynb`

### Models

| Model | Type | Notes |
|---|---|---|
| Finite Differencing | Naive baseline | Extrapolates from the last two readings |
| Linear Regression | Classical ML | Uses all engineered features + lag terms |
| ARIMA(1,1,1) | Time series | Univariate; trained on a single building/meter series |
| LSTM | Deep Learning | 2-layer LSTM in PyTorch; 24-hour sliding window input |

### Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

Both computed on the log-scale target (`log1p(meter_reading)`).

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels torch
```

---
