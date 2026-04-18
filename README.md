# DiamondIQ — Explainable AI-Powered Diamond Valuation System

A production-grade diamond price prediction and trend simulation platform built with Python, Streamlit, and XGBoost.

## Live App

[DiamondIQ on Streamlit Cloud](https://ashwin492-diamondpricepredictor-app-nfkevz.streamlit.app/)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Information](#model-information)
- [Preprocessing Pipeline](#preprocessing-pipeline)

## Overview

DiamondIQ goes beyond a basic price prediction tool. It provides an interactive valuation intelligence platform where users can predict diamond prices **and** simulate how price changes as carat weight varies — all powered by a tuned XGBoost model with a 98.1% R² score.

## Features

### Module A — Diamond Price Prediction Engine
- Structured input form (carat, cut, color, clarity, depth, table, x, y, z)
- Full preprocessing pipeline applied on every prediction (imputation → ordinal encoding → scaling)
- Premium price output card displayed instantly after submission

### Module B — Price Trend Simulation
- Interactive carat range slider (0.2 ct → 5.0 ct)
- Runs model inference across the full carat range while keeping all other features constant
- Live line chart: Carat (X-axis) vs Predicted Price (Y-axis)
- Metrics row: price at range min, your diamond, price at range max with % delta
- Auto-insight text when price acceleration is significant

### Data Visualization Page
Six interactive Plotly charts exploring the diamond dataset:
1. Price vs Carat by Cut Quality
2. Price Distribution by Cut
3. Correlation Heatmap of Numeric Features
4. Price by Color and Clarity
5. Carat Weight Distribution
6. Price Trends by Cut with OLS Trendlines

## Installation

### Prerequisites
- Python 3.x
- pip

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diamond-price-prediction.git
   cd diamond-price-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   # Windows
   myenv\Scripts\activate
   # macOS / Linux
   source myenv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Retrain the model:
   ```bash
   python training_pipeline.py
   ```

5. Launch the app:
   ```bash
   streamlit run app.py
   ```

## File Structure

```
diamondPricePredictor/
├── app.py                        # Streamlit app (prediction + simulation + visualizations)
├── training_pipeline.py          # End-to-end ML training pipeline
├── score.py                      # Azure ML scoring endpoint
├── main.py                       # Legacy Flask interface
├── requirements.txt
├── setup.py
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Loads data, creates train/test split
│   │   ├── data_tranformation.py # Preprocessing pipeline (impute → encode → scale)
│   │   └── model_trainer.py      # Optuna hyperparameter tuning across 6 models
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── artifacts/
│   ├── model.pkl                 # Trained XGBoost model
│   ├── preprocessor.pkl          # Fitted ColumnTransformer pipeline
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
├── notebook/
│   ├── data/gemstone.csv         # Source dataset
│   ├── EDA.ipynb
│   └── model_training.ipynb
│
└── templates/                    # Flask HTML templates (legacy)
```

## Model Information

Six models were evaluated using Optuna hyperparameter tuning (50 trials each, 5-fold CV):

| Model | R² Score |
|---|---|
| Linear Regression | 0.9078 |
| Lasso | 0.9078 |
| Ridge | 0.9078 |
| ElasticNet | 0.9078 |
| Decision Tree | 0.9715 |
| **XGBoost** | **0.9812** |

**Best model:** XGBoost — `max_depth=5`, `learning_rate=0.0413`, `n_estimators=283`

## Preprocessing Pipeline

| Feature Type | Steps |
|---|---|
| Numerical (carat, depth, table, x, y, z) | Median imputation → StandardScaler |
| Categorical (cut, color, clarity) | Most-frequent imputation → OrdinalEncoder → StandardScaler |

Ordinal category ordering:
- **Cut:** Fair → Good → Very Good → Premium → Ideal
- **Color:** D → E → F → G → H → I → J
- **Clarity:** I1 → SI2 → SI1 → VS2 → VS1 → VVS2 → VVS1 → IF
