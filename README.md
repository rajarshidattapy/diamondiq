# Diamond Price Prediction

This project is a web application that predicts the price of a diamond based on its characteristics using a machine learning model. The application is built using Python, with Streamlit as the web framework, and Scikit-learn for the machine learning components.

# You can use the application directly through the following link: [Diamond Price Predictor](https://ashwin492-diamondpricepredictor-app-nfkevz.streamlit.app/).

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build a regression model to predict the price of diamonds. The model uses various features such as carat weight, cut, color, clarity, depth, table, and dimensions (x, y, z) of the diamond. The app allows users to input these features and predicts the price of the diamond in real-time.

## Features
- **Real-time Price Prediction:** Users can input diamond characteristics and get a price prediction instantly.
- **Preprocessing Pipeline:** The app uses a preprocessing pipeline that handles missing values, encoding categorical features, and scaling numerical features.
- **Model Training:** The application trains several machine learning models and selects the best-performing model based on R² score.
- **User-Friendly Interface:** The app is built using Streamlit, making it easy to use and interact with.
- **Data Visualization:** The app provides visualizations to help users understand the relationships between diamond features and prices, including scatter plots, box plots, and correlation heatmaps.

## Installation

### Prerequisites
- Python 3.x
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diamond-price-prediction.git
2. Navigate to the project directory
   cd diamond-price-prediction
3. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
4. Install the required packages:
   pip install -r requirements.txt
5. Run the Streamlit app
   Run the Streamlit app

## File Structure
    ├── app.py                 # The main Streamlit app file
    ├── src/
    │   ├── components/
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   ├── model_trainer.py
    │   ├── logger.py
    │   ├── exception.py
    │   ├── utils.py
    ├── artifacts/             # Directory for storing artifacts (e.g., models, preprocessors)
    ├── notebooks/             # Jupyter notebooks for experimentation
    ├── training_pipeline.py   # Script for training the machine learning model
    ├── prediction_pipeline.py # Script for making predictions with the trained model
    ├── requirements.txt       # List of required Python packages
    └── README.md              # This file

## Model Information
The model is trained on a dataset of diamonds with various features. It uses the following machine learning models:

1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. ElasticNet Regression
5. Decision Tree Regressor  
6. XGBRegressor

### Model Training Output
During the model training process, the following output was observed:

- **Linear Regression**
  - `[2024-09-25 00:35:04,909] 71 root - INFO - Starting Optuna study for LinearRegression`
  - `[2024-09-25 00:35:06,800] 76 root - INFO - Best trial for LinearRegression: {} with R2 score: 0.9077971467453405`

- **Lasso**
  - `[2024-09-25 00:35:06,801] 71 root - INFO - Starting Optuna study for Lasso`
  - `[2024-09-25 00:36:48,367] 76 root - INFO - Best trial for Lasso: {'alpha': 0.12466789931248218} with R2 score: 0.9077908859489175`

- **Ridge**
  - `[2024-09-25 00:36:48,367] 71 root - INFO - Starting Optuna study for Ridge`
  - `[2024-09-25 00:36:49,670] 76 root - INFO - Best trial for Ridge: {'alpha': 0.9877354284937806} with R2 score: 0.9077992549637381`

- **ElasticNet**
  - `[2024-09-25 00:36:49,670] 71 root - INFO - Starting Optuna study for ElasticNet`
  - `[2024-09-25 00:38:26,954] 76 root - INFO - Best trial for ElasticNet: {'alpha': 0.00038093059857817273, 'l1_ratio': 0.9142385116456486} with R2 score: 0.9077899769121072`

- **Decision Tree**
  - `[2024-09-25 00:38:26,954] 71 root - INFO - Starting Optuna study for DecisionTree`
  - `[2024-09-25 00:38:37,565] 76 root - INFO - Best trial for DecisionTree: {'max_depth': 10, 'min_samples_split': 18} with R2 score: 0.971463449713504`

- **XGBoost**
  - `[2024-09-25 00:38:37,565] 71 root - INFO - Starting Optuna study for XGBoost`
  - `[2024-09-25 01:20:05,807] 76 root - INFO - Best trial for XGBoost: {'max_depth': 5, 'learning_rate': 0.041326860912172386, 'n_estimators': 283} with R2 score: 0.9812234655086796`

- **Model Report**
  - `[2024-09-25 01:20:05,809] 79 root - INFO - Model report: {'LinearRegression': 0.9077971467453405, 'Lasso': 0.9077908859489175, 'Ridge': 0.9077992549637381, 'ElasticNet': 0.9077899769121072, 'DecisionTree': 0.971463449713504, 'XGBoost': 0.9812234655086796}`
  - `[2024-09-25 01:20:05,809] 99 root - INFO - Best Model Found, Model name: XGBoost, R2 score: 0.9812234655086796`




## Preprocessing Pipeline
The preprocessing pipeline includes:

1. Numerical Features: Imputation of missing values using the median and scaling using StandardScaler.
2. Categorical Features: Imputation of missing values using the most frequent strategy, ordinal encoding, and scaling.

## Data Visualization
The application includes several visualizations to explore the diamond dataset, including:

1. Price vs Carat: A scatter plot showing the relationship between a diamond's carat weight and its price.
2. Price Distribution by Cut: A box plot displaying the distribution of diamond prices for each cut quality.
3. Correlation Heatmap: A heatmap showing correlations between numeric features.
4. Price by Color and Clarity: A box plot showing how diamond prices vary across different color grades and clarity levels.
5. Carat Distribution: A histogram displaying the distribution of diamond carat weights.
6. Price Trends by Cut and Carat: A scatter plot with trend lines showing how the relationship between carat weight and price varies across different cut qualities.
