import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="DiamondIQ", page_icon="💎", layout="wide")

@st.cache_resource
def load_objects():
    preprocessor = joblib.load(os.path.join('artifacts', 'preprocessor.pkl'))
    model = joblib.load(os.path.join('artifacts', 'model.pkl'))
    return preprocessor, model

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join('artifacts', 'raw.csv'))

preprocessor, model = load_objects()
data = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("💎 DiamondIQ")
page = st.sidebar.selectbox("Navigate", ["Prediction", "Visualization"])
st.sidebar.divider()
st.sidebar.caption("Explainable AI-Powered Diamond Valuation")
st.sidebar.markdown("""
**Features**
- Carat · Cut · Color · Clarity
- Depth · Table · X · Y · Z
""")
st.sidebar.header("Model Info")
st.sidebar.text(f"Preprocessor: {type(preprocessor).__name__}")
st.sidebar.text(f"Model: {type(model).__name__}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_price(inputs: dict) -> float:
    df = pd.DataFrame([inputs])
    processed = preprocessor.transform(df)
    return float(model.predict(processed)[0])

def run_carat_simulation(inputs: dict, carat_values: np.ndarray) -> np.ndarray:
    rows = []
    for c in carat_values:
        row = inputs.copy()
        row['carat'] = c
        rows.append(row)
    df = pd.DataFrame(rows)
    processed = preprocessor.transform(df)
    return model.predict(processed)

# ── Prediction Page ───────────────────────────────────────────────────────────
if page == "Prediction":
    st.title("💎 DiamondIQ — Valuation Engine")
    st.markdown("Enter the diamond's characteristics and click **Predict Price** to get an AI-powered valuation.")

    # Session state init
    if 'predicted' not in st.session_state:
        st.session_state.predicted = False
        st.session_state.prediction = None
        st.session_state.inputs = {}

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("Diamond Characteristics")
        col1, col2, col3 = st.columns(3)

        with col1:
            carat   = st.number_input("Carat Weight",  min_value=0.1, max_value=10.0, value=1.0,  step=0.01)
            cut     = st.selectbox("Cut",     ["Fair", "Good", "Very Good", "Premium", "Ideal"])
            color   = st.selectbox("Color",   ["D", "E", "F", "G", "H", "I", "J"])

        with col2:
            clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
            depth   = st.number_input("Depth %",       min_value=0.1, max_value=100.0, value=61.5, step=0.1)
            table   = st.number_input("Table %",       min_value=0.1, max_value=100.0, value=57.0, step=0.1)

        with col3:
            x = st.number_input("Length X (mm)", min_value=0.1, max_value=100.0, value=5.0, step=0.01)
            y = st.number_input("Width  Y (mm)", min_value=0.1, max_value=100.0, value=5.0, step=0.01)
            z = st.number_input("Depth  Z (mm)", min_value=0.1, max_value=100.0, value=3.1, step=0.01)

        submitted = st.form_submit_button("Predict Price", use_container_width=True, type="primary")

    if submitted:
        inputs = dict(carat=carat, cut=cut, color=color, clarity=clarity,
                      depth=depth, table=table, x=x, y=y, z=z)
        try:
            price = predict_price(inputs)
            st.session_state.predicted  = True
            st.session_state.prediction = price
            st.session_state.inputs     = inputs
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state.predicted = False

    # ── Prediction Card ───────────────────────────────────────────────────────
    if st.session_state.predicted:
        price  = st.session_state.prediction
        inputs = st.session_state.inputs

        st.markdown("---")
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    border-radius: 16px;
                    padding: 40px 30px;
                    text-align: center;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
                    border: 1px solid rgba(255,255,255,0.08);
                ">
                    <p style="color:#a0aec0;font-size:13px;letter-spacing:2px;
                              text-transform:uppercase;margin-bottom:6px;">
                        Predicted Diamond Price
                    </p>
                    <h1 style="color:#f6e05e;font-size:52px;font-weight:800;margin:0;">
                        ${price:,.0f}
                    </h1>
                    <p style="color:#68d391;font-size:13px;margin-top:14px;">
                        ✓ {inputs['carat']} ct &nbsp;·&nbsp; {inputs['cut']} &nbsp;·&nbsp;
                        {inputs['color']} &nbsp;·&nbsp; {inputs['clarity']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Price Trend Simulation ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 Price Trend Simulation")
        st.markdown(
            "Slide the carat range to simulate how price evolves while keeping "
            "all other characteristics constant."
        )

        carat_range = st.slider(
            "Select Carat Range",
            min_value=0.2,
            max_value=5.0,
            value=(0.5, 2.5),
            step=0.1,
            format="%.1f ct",
        )
        carat_min, carat_max = carat_range
        carat_values = np.round(np.arange(carat_min, carat_max + 0.05, 0.1), 2)

        sim_prices = run_carat_simulation(inputs, carat_values)

        # Line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=carat_values,
            y=sim_prices,
            mode="lines+markers",
            line=dict(color="#f6e05e", width=3),
            marker=dict(size=5, color="#f6e05e"),
            fill="tozeroy",
            fillcolor="rgba(246,224,94,0.07)",
            name="Predicted Price",
        ))
        fig.add_trace(go.Scatter(
            x=[inputs['carat']],
            y=[price],
            mode="markers",
            marker=dict(size=14, color="#fc8181", symbol="star"),
            name="Your Diamond",
        ))
        fig.update_layout(
            title=f"Carat vs Price  ·  {inputs['cut']} cut · {inputs['color']} color · {inputs['clarity']} clarity",
            xaxis_title="Carat Weight",
            yaxis_title="Predicted Price (USD)",
            template="plotly_dark",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics row
        price_at_min  = float(sim_prices[0])
        price_at_max  = float(sim_prices[-1])
        pct_change    = (price_at_max - price_at_min) / price_at_min * 100 if price_at_min else 0

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Price at {carat_min:.1f} ct", f"${price_at_min:,.0f}")
        m2.metric("Your Diamond",                  f"${price:,.0f}", f"{inputs['carat']} ct")
        m3.metric(f"Price at {carat_max:.1f} ct",  f"${price_at_max:,.0f}", f"+{pct_change:.0f}%")

        # Insight
        mid = len(sim_prices) // 2
        second_half_avg = sim_prices[mid:].mean()
        first_half_avg  = sim_prices[:mid].mean()
        if second_half_avg - first_half_avg > 2000:
            inflection = carat_values[mid]
            st.info(f"Price accelerates sharply after **{inflection:.1f} carat** for this diamond grade.")

# ── Visualization Page ────────────────────────────────────────────────────────
elif page == "Visualization":
    st.title("📊 Diamond Data Insights")

    st.subheader("1. Price vs Carat by Cut Quality")
    fig1 = px.scatter(data, x="carat", y="price", color="cut",
                      hover_data=["clarity", "color"],
                      title="Diamond Price vs Carat")
    fig1.update_layout(xaxis_title="Carat", yaxis_title="Price ($)")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
- Strong positive correlation between carat and price.
- Higher quality cuts (Ideal, Premium) command higher prices at the same carat weight.
- Price variance increases for larger diamonds, where cut/color/clarity matter more.
""")

    st.subheader("2. Price Distribution by Cut")
    fig2 = px.box(data, x="cut", y="price", color="cut",
                  title="Price Distribution by Cut")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
- Ideal and Premium cuts have higher median prices and wider ranges.
- Fair cuts show the lowest median and narrowest spread.
- Significant overlap across cuts confirms that other factors also drive price.
""")

    st.subheader("3. Correlation Heatmap")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr = data[numeric_cols].corr()
    fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                     title="Correlation Heatmap of Numeric Features")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
- Carat has the strongest positive correlation with price.
- Dimensions (x, y, z) are highly correlated with each other and with carat.
- Depth and table percentages show weak correlations with price.
""")

    st.subheader("4. Price by Color and Clarity")
    fig4 = px.box(data, x="color", y="price", color="clarity",
                  title="Price Distribution by Color and Clarity")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""
- Prices generally decrease from D (best) to J (worst) color grade.
- Within each color grade, better clarity (I1 → IF) commands higher prices.
- Color impact is more pronounced at higher clarity grades.
""")

    st.subheader("5. Carat Weight Distribution")
    fig5 = px.histogram(data, x="carat", nbins=50,
                        title="Distribution of Diamond Carat Weights")
    fig5.update_layout(bargap=0.1)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""
- Right-skewed distribution: most diamonds are between 0.3 and 1.5 carats.
- Peaks at round carat values (0.5, 1.0, 1.5) reflect consumer preferences.
- Diamonds over 2 carats are relatively rare.
""")

    st.subheader("6. Price Trends by Cut and Carat")
    fig6 = px.scatter(data, x="carat", y="price", color="cut", trendline="ols",
                      title="Price Trends by Cut and Carat")
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
- All cut qualities show a positive linear relationship between carat and price.
- Price-per-carat rises faster for higher quality cuts (Ideal, Premium).
- The price gap between cut qualities widens for larger diamonds.
""")
