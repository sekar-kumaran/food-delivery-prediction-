# 🛵 Food Delivery Time Prediction

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A complete end-to-end Machine Learning project to predict food delivery times based on real-world Indian delivery dataset. It includes data exploration, model training, and a stunning, interactive web application built with Streamlit.

### 🌐 Live Demo: [https://food-delivery-model.streamlit.app/](https://food-delivery-model.streamlit.app/)
## 🌟 Features
- **Stunning UI**: The application features a custom CSS glassmorphism design, animated gradient backgrounds, and fully responsive elements.
- **Dataset Overview**: Real-time data preview directly integrated into the app.
- **Model Predictions**: Uses a trained **Random Forest Regressor** to predict the estimated delivery time based on parameters like distance, rider age, weather, traffic, and more.
- **Model Evaluation**: Transparent metrics showing why Random Forest was chosen over Linear Regression and Decision Trees (achieved an R² Score of 80.5%).

## 🛠️ Tech Stack
- **Data Science**: Python, Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest, Decision Trees, Linear Regression)
- **Web App**: Streamlit
- **Model Serialization**: Joblib

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/sekar-kumaran/food-delivery-prediction-.git
   cd food-delivery-prediction-
   ```

2. **Install the dependencies**
   Ensure you have Python installed. Then run:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

3. **Train the Model (If applicable)**
   *Note: Pre-trained `.joblib` models are excluded from this repo due to size limits. You must run the Jupyter notebook to generate them!*
   - Open `Food_Delivery_Time_Prediction_Project.ipynb`.
   - Run all cells to process the data and generate `best_delivery_time_model.joblib`.

4. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your browser at `http://localhost:8502`.

## 📊 Dataset Details
The model analyzes the following features:
- **Delivery Person Info:** Age and average rating.
- **Distance:** Calculated using Restaurant and Delivery geographic coordinates.
- **Environment:** Current weather (Fog, Stormy, Sunny, etc.) and Road Traffic Density.
- **Vehicle Type:** Scooter, Motorcycle, Bicycle, etc.
- **External Factors:** City type, Multiple Deliveries, and Festival occurrences.

## 📈 Performance
- **Algorithm:** Random Forest Regressor
- **R² Score:** 0.805
- **MAE:** 3.22 mins
- **RMSE:** 4.10 mins
