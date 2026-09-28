import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a stunning Glassmorphism UI and enforcing Dark Text for Light Theme
page_bg_css = """
<style>
/* Stunning animated background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #f3e7e9, #e3eeff, #f3e7e9, #e3eeff);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Sidebar Gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f0f4f8 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.5);
}

/* Typography Enhancements - Force dark text */
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
p, span, label, li, b, strong, div, .stMarkdown, .stText {
    color: #34495e !important;
}

/* Explicit exception to keep the result card text white */
.result-card, .result-card h1, .result-card h2, .result-card span {
    color: white !important;
}

/* Glassmorphism Info Box */
.glass-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    margin-bottom: 25px;
}

/* Clean UI Lists */
.custom-list {
    list-style-type: none;
    padding-left: 0;
}
.custom-list li {
    background: rgba(255, 255, 255, 0.5);
    margin: 8px 0;
    padding: 12px 18px;
    border-radius: 8px;
    border-left: 4px solid #4a90e2;
    transition: transform 0.2s ease;
}
.custom-list li:hover {
    transform: translateX(5px);
    background: rgba(255, 255, 255, 0.8);
}

/* Metric Cards */
.metric-container {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
}
.metric-card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    width: 30%;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border-bottom: 5px solid #4a90e2;
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
.metric-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #7f8c8d !important;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #2c3e50 !important;
}

/* Primary Button Styling - Exception for text color */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 30px;
    padding: 12px 30px;
    font-weight: bold;
    font-size: 1.1rem;
    border: none;
    box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
    transition: all 0.3s ease;
    width: 100%;
}
.stButton>button * {
    color: white !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    box-shadow: 0 6px 20px rgba(118, 75, 162, 0.6);
    transform: scale(1.02);
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# Cache model loading for speed
@st.cache_resource
def load_model():
    model_dict = joblib.load('best_delivery_time_model_compressed.joblib')
    return model_dict['model'], model_dict['columns']

@st.cache_data
def load_data():
    df = pd.read_json('India-Food-Delivery-Time-Prediction.json')
    return df.head(100) # Load subset for preview

try:
    model, expected_columns = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("App Navigation")
st.sidebar.markdown("Navigate through the application to explore data and make predictions.")
page = st.sidebar.radio("", ["📊 Dataset Overview", "🛵 Predict Delivery Time", "📈 Model Evaluation"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the prediction page to estimate real-world delivery times using our Random Forest algorithm.")

# --- PAGE 1: DATASET OVERVIEW ---
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    
    html_overview = """<div class="glass-card">
<h2>About the Food Delivery Dataset</h2>
<p style="font-size: 1.1rem; line-height: 1.6;">
This application utilizes historical data collected from various food deliveries across India. 
The primary objective of the machine learning model is to accurately predict the <b>Time Taken (in minutes)</b> 
for a delivery to reach the customer based on real-time factors.
</p>
<h3 style="margin-top: 20px;">Key Features Analyzed</h3>
<ul class="custom-list">
<li>🧑‍🛵 <b>Delivery Person Details:</b> Age and average customer ratings of the rider.</li>
<li>📍 <b>Location & Distance:</b> The calculated geographic distance (km) between the restaurant and the delivery point.</li>
<li>🌦️ <b>Environment:</b> Weather conditions (Sunny, Stormy, Fog) and Road traffic density (Low, Jam, High).</li>
<li>🛵 <b>Vehicle & Order:</b> Type of vehicle used, its condition, and the category of the food order.</li>
<li>🎉 <b>External Factors:</b> City type (Urban/Metropolitian) and whether it's a Festival day.</li>
</ul>
</div>"""
    st.markdown(html_overview, unsafe_allow_html=True)

    st.subheader("Data Preview (First 100 Rows)")
    with st.spinner("Loading dataset preview..."):
        try:
            df_preview = load_data()
            st.dataframe(df_preview, use_container_width=True, height=400)
        except Exception as e:
            st.warning("Could not load dataset preview.")


# --- PAGE 2: MODEL PREDICTION ---
elif page == "🛵 Predict Delivery Time":
    st.title("🛵 Delivery Time Predictor")
    st.markdown('<p style="font-size: 1.2rem;">Tweak the parameters below to see how different factors affect the estimated delivery time!</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧑 Rider Info")
        age = st.slider("Age", min_value=18, max_value=65, value=30)
        ratings = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        
    with col2:
        st.markdown("### 🛣️ Route & Traffic")
        distance = st.number_input("Distance (km)", min_value=1.0, max_value=50.0, value=5.0, step=0.5)
        traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
        
    with col3:
        st.markdown("### 📦 Order Details")
        vehicle_cond = st.selectbox("Vehicle Condition (0-2)", [0, 1, 2], index=2)
        vehicle_type = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
        order_type = st.selectbox("Food Category", ["Snack", "Meal", "Drinks", "Buffet"])

    st.markdown("---")
    
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### 🌦️ Environment")
        weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"])

    with col5:
        st.markdown("### 🎒 Workload")
        multiple_deliveries = st.selectbox("Multiple Deliveries?", [0, 1, 2, 3])
        festival = st.radio("Is it a Festival?", ["No", "Yes"])

    with col6:
        st.markdown("### 🏙️ Location")
        city = st.selectbox("City Type", ["Urban", "Semi-Urban", "Metropolitian"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Button
    if st.button("🚀 Calculate Estimated Delivery Time"):
        with st.spinner("Analyzing data and generating prediction..."):
            # Prepare input data
            input_data = {
                'Delivery_person_Age': age,
                'Delivery_person_Ratings': ratings,
                'distance_km': distance,
                'Vehicle_condition': vehicle_cond,
                'multiple_deliveries': multiple_deliveries
            }
            
            for col in expected_columns:
                if col not in input_data:
                    input_data[col] = 0
                    
            # One-hot encoding
            if f'Weatherconditions_{weather}' in input_data: input_data[f'Weatherconditions_{weather}'] = 1
            if f'Road_traffic_density_{traffic}' in input_data: input_data[f'Road_traffic_density_{traffic}'] = 1
            if f'Type_of_order_{order_type}' in input_data: input_data[f'Type_of_order_{order_type}'] = 1
            if f'Type_of_vehicle_{vehicle_type}' in input_data: input_data[f'Type_of_vehicle_{vehicle_type}'] = 1
            if festival == "Yes": input_data['Festival_Yes'] = 1
            if f'City_{city}' in input_data: input_data[f'City_{city}'] = 1

            input_df = pd.DataFrame([input_data])[expected_columns]
            
            try:
                prediction = model.predict(input_df)[0]
                html_prediction = f"""<div class="result-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 25px; border-radius: 15px; text-align: center; margin-top: 20px; box-shadow: 0 10px 20px rgba(17,153,142,0.3);">
<h2 style="margin: 0;">⏱️ Estimated Delivery Time</h2>
<h1 style="font-size: 3.5rem; margin: 10px 0;">{prediction:.1f} <span style="font-size: 1.5rem;">minutes</span></h1>
</div>"""
                st.markdown(html_prediction, unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"An error occurred: {e}")


# --- PAGE 3: MODEL EVALUATION ---
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    
    html_evaluation = """<div class="glass-card">
<h2>Performance Metrics</h2>
<p style="font-size: 1.1rem; line-height: 1.6;">
We trained multiple regression models to find the most accurate algorithm for predicting delivery times. 
Below is a comparison of their performance on unseen test data.
</p>
</div>"""
    st.markdown(html_evaluation, unsafe_allow_html=True)
    
    st.markdown("### 🏆 Random Forest Regressor <span style='font-size: 1rem; color: #4CAF50 !important;'>(Currently Active Model)</span>", unsafe_allow_html=True)
    
    html_rf_metrics = """<div class="metric-container">
<div class="metric-card"><div class="metric-title">Mean Absolute Error (MAE)</div><div class="metric-value" style="color:#27ae60 !important;">3.22m</div></div>
<div class="metric-card"><div class="metric-title">Root Mean Squared Error</div><div class="metric-value" style="color:#27ae60 !important;">4.10m</div></div>
<div class="metric-card"><div class="metric-title">R² Score</div><div class="metric-value" style="color:#27ae60 !important;">80.5%</div></div>
</div>"""
    st.markdown(html_rf_metrics, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 🥈 Decision Tree Regressor")
    html_dt_metrics = """<div class="metric-container">
<div class="metric-card"><div class="metric-title">Mean Absolute Error (MAE)</div><div class="metric-value">3.35m</div></div>
<div class="metric-card"><div class="metric-title">Root Mean Squared Error</div><div class="metric-value">4.30m</div></div>
<div class="metric-card"><div class="metric-title">R² Score</div><div class="metric-value">78.5%</div></div>
</div>"""
    st.markdown(html_dt_metrics, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
        
    st.markdown("### 🥉 Linear Regression")
    html_lr_metrics = """<div class="metric-container">
<div class="metric-card"><div class="metric-title">Mean Absolute Error (MAE)</div><div class="metric-value" style="color:#c0392b !important;">4.76m</div></div>
<div class="metric-card"><div class="metric-title">Root Mean Squared Error</div><div class="metric-value" style="color:#c0392b !important;">6.02m</div></div>
<div class="metric-card"><div class="metric-title">R² Score</div><div class="metric-value" style="color:#c0392b !important;">57.9%</div></div>
</div>"""
    st.markdown(html_lr_metrics, unsafe_allow_html=True)
