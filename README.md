# MATH8800 — Project 2: Energy Consumption Forecasting

**Author:** Ze Yu Jiang  
**Date:** May 2026

---

## Overview

This project explores building energy consumption forecasting using the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction) dataset. It compares classical baseline methods against deep learning approaches (LSTM) for predicting hourly meter readings across multiple building types and energy meter types.

A second component (`NA_Project3.py`) implements the **Conjugate Gradient method** from scratch for solving tridiagonal linear systems, tested at multiple problem sizes and tolerance levels.

---

## Repository Structure

```
├── NA_Project3.py       # Conjugate Gradient solver (Numerical Analysis)
├── test.ipynb           # Energy forecasting notebook (main project)
├── .gitignore
└── data/                # Not included — see Data section below
```

---

## Data

The `data/` directory is excluded from this repository due to file size (the CSVs total ~2 GB).

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

### Pipeline

1. **Data Loading** — Load train, building metadata, and weather CSVs
2. **Preprocessing**
   - Impute missing `year_built` and `floor_count` with column medians
   - Encode `primary_use` as a numeric category
   - Interpolate missing weather values per site using linear interpolation
   - Remove known faulty readings (site 0, electricity meter, pre-2016-05-21)
3. **Feature Engineering**
   - Cyclic time encoding: `hour`, `dayofweek`, `month` → sin/cos pairs
   - `is_weekend` binary flag
   - Lag features: 1hr, 2hr, 24hr (same hour yesterday), 168hr (same hour last week)
4. **Target Transformation** — Apply `log1p` to compress the meter reading scale
5. **Scaling** — `StandardScaler` on continuous features
6. **Chronological Train/Val Split** — 80/20 split by timestamp (no data leakage)

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

## Script: `NA_Project3.py`

Implements the **Conjugate Gradient (CG)** iterative solver for tridiagonal systems $Ax = b$.

- Matrix $A$ is represented by three diagonals (sub, main, super) — no dense storage
- Tested on problem sizes: $n \in \{10, 100, 1000, 2000\}$
- Two matrix variants: diagonal value = 2 and 3
- Two tolerance levels: $10^{-8}$ and $10^{-16}$
- Reports iteration count, final residual norm, and first iteration below tolerance

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels torch
```

---
