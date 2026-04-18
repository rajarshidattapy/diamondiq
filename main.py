# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.utils
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
def load_data():
    try:
        return pd.read_csv('notebook/data/gemstone.csv')
    except FileNotFoundError:
        print("Data file not found. Creating a dummy dataset...")
        return create_dummy_dataset()

def create_dummy_dataset():
    # Create a small dummy dataset
    data = pd.DataFrame({
        'carat': np.random.uniform(0.2, 5.0, 1000),
        'cut': np.random.choice(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], 1000),
        'color': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], 1000),
        'clarity': np.random.choice(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], 1000),
        'depth': np.random.uniform(55, 70, 1000),
        'table': np.random.uniform(50, 70, 1000),
        'price': np.random.uniform(300, 20000, 1000),
        'x': np.random.uniform(3, 10, 1000),
        'y': np.random.uniform(3, 10, 1000),
        'z': np.random.uniform(2, 7, 1000)
    })
    return data

def create_and_fit_preprocessor_and_model():
    # Load or create the dataset
    data = load_data()
    
    # Define features and target
    features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    target = 'price'
    
    # Split the data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the preprocessing steps
    numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    categorical_features = ['cut', 'color', 'clarity']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create and fit the preprocessor
    preprocessor.fit(X_train)

    # Create and fit the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(preprocessor.transform(X_train), y_train)

    return preprocessor, model

# Always create and fit new preprocessor and model
preprocessor, model = create_and_fit_preprocessor_and_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input from form
        carat = float(request.form['carat'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

        # Create a dataframe from inputs
        input_data = pd.DataFrame({
            'carat': [carat],
            'cut': [cut],
            'color': [color],
            'clarity': [clarity],
            'depth': [depth],
            'table': [table],
            'x': [x],
            'y': [y],
            'z': [z]
        })

        try:
            # Preprocess the input
            input_processed = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_processed)
            
            # Format the prediction
            prediction = f"${prediction[0]:,.2f}"
            
            return render_template('predict.html', prediction=prediction)
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/visualize')
def visualize():
    # Load data for visualization
    data = load_data()

    # 1. Price vs Carat with Cut as Color
    fig_scatter = px.scatter(data, x='carat', y='price', color='cut', hover_data=['clarity', 'color'],
                             title='Diamond Price vs Carat')
    fig_scatter.update_layout(xaxis_title='Carat', yaxis_title='Price ($)')
    scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Price Distribution by Cut
    fig_box = px.box(data, x='cut', y='price', color='cut', title="Price Distribution by Cut")
    box_json = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Correlation Heatmap
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Correlation Heatmap of Numeric Features')
    corr_json = json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder)

    # 4. Price by Color and Clarity
    fig_color_clarity = px.box(data, x='color', y='price', color='clarity', 
                               title='Price Distribution by Color and Clarity')
    color_clarity_json = json.dumps(fig_color_clarity, cls=plotly.utils.PlotlyJSONEncoder)

    # 5. Carat Distribution
    fig_carat_dist = px.histogram(data, x='carat', nbins=50, title='Distribution of Diamond Carat Weights')
    fig_carat_dist.update_layout(bargap=0.1)
    carat_dist_json = json.dumps(fig_carat_dist, cls=plotly.utils.PlotlyJSONEncoder)

    # 6. Price Trends by Cut and Carat
    fig_trend = px.scatter(data, x='carat', y='price', color='cut', trendline='ols',
                           title='Price Trends by Cut and Carat')
    trend_json = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('visualize.html', 
                           scatter_json=scatter_json,
                           box_json=box_json,
                           corr_json=corr_json,
                           color_clarity_json=color_clarity_json,
                           carat_dist_json=carat_dist_json,
                           trend_json=trend_json)

if __name__ == '__main__':
    app.run(debug=True)