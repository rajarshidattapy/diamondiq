import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import numpy as np
import joblib

# Load the preprocessor and model
def load_objects():
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
    model_path = os.path.join('artifacts', 'model.pkl')
    
    # Load preprocessor and model using Joblib
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    
    return preprocessor, model

preprocessor, model = load_objects()

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('notebook/data/gemstone.csv')

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Visualization"])

if page == "Prediction":
    # Prediction Page
    st.title('Diamond Price Prediction')

    # Input fields
    st.header('Enter Diamond Characteristics:')

    carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
    cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox('Color Rating:', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.number_input('Depth Percentage:', min_value=0.1, max_value=100.0, value=60.0)
    table = st.number_input('Table Percentage:', min_value=0.1, max_value=100.0, value=55.0)
    x = st.number_input('Length (X) in mm:', min_value=0.1, max_value=100.0, value=5.0)
    y = st.number_input('Width (Y) in mm:', min_value=0.1, max_value=100.0, value=5.0)
    z = st.number_input('Depth (Z) in mm:', min_value=0.1, max_value=100.0, value=3.0)

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

    # Predict button
    if st.button('Predict Price'):
        try:
            # Preprocess the input
            input_processed = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_processed)
            
            # Display the result
            st.success(f'The predicted price of the diamond is ${prediction[0]:,.2f}')
        except Exception as e:
            st.error(f"Error in prediction: {e}")

elif page == "Visualization":
    # Visualization Page
    st.title('Diamond Data Visualizations')

    # 1. Price vs Carat with Cut as Color
    st.subheader('1. Price vs Carat with Cut as Color')
    fig_scatter = px.scatter(data, x='carat', y='price', color='cut', hover_data=['clarity', 'color'],
                             title='Diamond Price vs Carat')
    fig_scatter.update_layout(xaxis_title='Carat', yaxis_title='Price ($)')
    st.plotly_chart(fig_scatter)
    st.markdown("""This scatter plot shows the relationship between a diamond's carat weight and its price, with the cut quality represented by color.
    - There's a strong positive correlation between carat weight and price.
    - Higher quality cuts (Ideal, Premium) tend to be more expensive for the same carat weight.
    - Price variation increases with carat weight, suggesting other factors (like cut, color, and clarity) have more influence on price for larger diamonds.
    """)

    # 2. Price Distribution by Cut
    st.subheader('2. Price Distribution by Cut')
    fig_box = px.box(data, x='cut', y='price', color='cut', title="Price Distribution by Cut")
    st.plotly_chart(fig_box)
    st.markdown("""This box plot displays the distribution of diamond prices for each cut quality.
    - Ideal and Premium cuts generally have higher median prices and wider price ranges.
    - Fair cuts have the lowest median price and the narrowest price range.
    - There's significant overlap in price ranges across all cut qualities, indicating that other factors also influence price.
    """)

    # 3. Correlation Heatmap (only for numeric columns)
    st.subheader('3. Correlation Heatmap (Numeric Features)')
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Correlation Heatmap of Numeric Features')
    st.plotly_chart(fig_corr)
    st.markdown("""This heatmap shows the correlation between numeric features in the dataset.
    - Carat has the strongest positive correlation with price.
    - The dimensions (x, y, z) are highly correlated with each other and with carat weight.
    - Depth and table percentages have weak correlations with price, suggesting they're less important in determining a diamond's value.
    """)

    # 4. Price by Color and Clarity
    st.subheader('4. Price by Color and Clarity')
    fig_color_clarity = px.box(data, x='color', y='price', color='clarity', 
                               title='Price Distribution by Color and Clarity')
    st.plotly_chart(fig_color_clarity)
    st.markdown("""This box plot shows how diamond prices vary across different color grades and clarity levels.
    - Generally, prices decrease as we move from D (best) to J (worst) in color.
    - Within each color grade, there's a clear trend of increasing prices with better clarity (from I1 to IF).
    - The impact of color on price seems more pronounced for higher clarity grades.
    """)

    # 5. Carat Distribution
    st.subheader('5. Carat Distribution')
    fig_carat_dist = px.histogram(data, x='carat', nbins=50, title='Distribution of Diamond Carat Weights')
    fig_carat_dist.update_layout(bargap=0.1)
    st.plotly_chart(fig_carat_dist)
    st.markdown("""This histogram shows the distribution of diamond carat weights in the dataset.
    - The distribution is right-skewed, with most diamonds weighing between 0.3 and 1.5 carats.
    - There are noticeable peaks at round numbers (0.5, 1.0, 1.5 carats), suggesting pricing strategies or consumer preferences.
    - Diamonds over 2 carats are relatively rare in this dataset.
    """)

    # 6. Price Trends by Cut and Carat
    st.subheader('6. Price Trends by Cut and Carat')
    fig_trend = px.scatter(data, x='carat', y='price', color='cut', trendline='ols',
                           title='Price Trends by Cut and Carat')
    st.plotly_chart(fig_trend)
    st.markdown("""This scatter plot with trend lines shows how the relationship between carat weight and price varies across different cut qualities.
    - All cut qualities show a positive linear relationship between carat and price.
    - The slope of the trend lines suggests that price increases more rapidly with carat weight for higher quality cuts (Ideal, Premium).
    - There's significant variation around the trend lines, indicating that other factors also influence price.
    - The difference in price between cut qualities becomes more pronounced for larger diamonds.
    """)

# Sidebar information
st.sidebar.header('About')
st.sidebar.info('This app predicts the price of a diamond based on its characteristics using a machine learning model.')
st.sidebar.header('Features Used')
st.sidebar.markdown("""- Carat Weight: Weight of the diamond
- Cut: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- Color: Diamond color, from J (worst) to D (best)
- Clarity: A measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- Depth: Total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
- Table: Width of top of diamond relative to widest point (43--95)
- X: Length in mm (0--10.74)
- Y: Width in mm (0--58.9)
- Z: Depth in mm (0--31.8)
""")

# Display preprocessor and model information
st.sidebar.header('Model Information')
st.sidebar.text(f"Preprocessor type: {type(preprocessor).__name__}")
st.sidebar.text(f"Model type: {type(model).__name__}")
