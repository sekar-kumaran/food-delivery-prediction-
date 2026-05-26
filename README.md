# Food Delivery Time Prediction Project

This repository contains a comprehensive workflow for predicting food delivery times using a real-world dataset of orders from Indian cities. The project covers the full pipeline of data cleaning, exploratory data analysis (EDA), feature engineering, and supervised machine learning modeling.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Data Processing Workflow](#data-processing-workflow)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Results & Insights](#results--insights)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [References](#references)

---

## Project Overview

The main goal is to predict the time required to deliver food orders, given various influencing factors. Having accurate delivery time predictions can help restaurants, delivery companies, and customers optimize planning and expectations.

**Key objectives:**
- Analyze the data to reveal main drivers of delivery time.
- Build regression models to predict delivery duration.
- Evaluate and compare model performance.

---

## Dataset Description

**Source:**  
- `India-Food-Delivery-Time-Prediction.json` (included in the repo).

**Features include:**
- Delivery agent ID, Age, and Ratings
- Restaurant & delivery locations (latitude/longitude)
- Order date, time ordered, and picked up
- Weather & road traffic at time of order
- Vehicle condition, type of order, festival flag, city
- Target variable: *Time_taken(min)*

**Dimensions:**
- ~42,000 rows (orders)
- 20 columns

**Sample columns:**
| Feature                      | Example                                  |
|------------------------------|------------------------------------------|
| Delivery_person_Age          | 34.0                                     |
| Delivery_person_Ratings      | 4.6                                      |
| Road_traffic_density         | High, Jam, Low, Medium                   |
| Weatherconditions            | Sunny, Cloudy, Stormy, Sandstorms        |
| Type_of_vehicle              | motorcycle, scooter                      |
| Distance (engineered)        | 3.0 km (from Haversine formula)          |
| Time_taken(min)              | 24, 30, 33, ...                          |

---

## Data Processing Workflow

**1. Data Loading**
- Data is loaded from the JSON file using pandas.

**2. Cleaning**
- Fix inconsistent missing values (`NaN`, `nan` as strings) to proper nulls.
- Strip spaces in text columns.
- Remove redundant text in columns (e.g., remove ‘(min)’ from delivery time).
- Fill missing numeric values with medians.
- Fill missing categorical values with the mode (most common value).
- Standardize categorical variable values.

**3. Feature Engineering**
- Calculate the distance between restaurant and delivery location using Haversine formula.
- Convert relevant features to correct data types.

---

## Exploratory Data Analysis (EDA)

- **Distribution Analysis:**  
  Histogram plot of delivery times for all orders.
- **Traffic Impact:**  
  Countplot of orders across different road traffic conditions.
- **Weather Impact:**  
  Boxplot showing effect of weather on delivery time.
- **Distance Effect:**  
  Scatterplot of trip distance vs delivery time.

**Example Insight Plots:**
- Delivery times are higher on days with stormy weather and heavy/jam traffic.
- Longer distances generally require more time, but variability can depend on traffic and weather.

---

## Feature Engineering

- **Distance Calculation:**  
  The distance between restaurant and delivery address is computed for each order using the latitude/longitude values and the Haversine formula.

---

## Machine Learning Models

**Regression models evaluated:**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**Modeling Pipeline:**
1. Data split into training and testing sets.
2. Features and target variable selected.
3. Multiple regression models fitted and evaluated.
4. Performance metrics used for comparison:  
   - Mean Absolute Error (MAE)  
   - Root Mean Squared Error (RMSE)  
   - R² Score

**Example model usage:**
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

---

## Results & Insights

- **Best Model:** Random Forest yields the most accurate predictions.
- **Importance Factors:**  
  - Traffic density, weather, and route distance have the most significant impact on time.
- **Typical MAE:**  
  - (Example) MAE for Random Forest: ~3.2 minutes (change based on your actual result).
- **Feature Insights:**  
  - Late orders cluster around poor traffic and high festival periods.
- **Visual Proof:**
  
  ![Histogram](#) *(Add .png, if desired, from notebook output)*

---

## How to Run

1. Clone/download this repository.
2. Ensure the dataset (`India-Food-Delivery-Time-Prediction.json`) is present in the repo root.
3. Open and step through the notebook:  
   [`Food_Delivery_Time_Prediction_Project.ipynb`](https://github.com/sekar-kumaran461/food-delivery-prediction-/blob/main/Food_Delivery_Time_Prediction_Project.ipynb)
4. Install required libraries with:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

---

## Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn, joblib

---

## References

- Data and notebook authored by [sekar-kumaran461](https://github.com/sekar-kumaran461)
- See notebook for further credits and cited sources.

---

*For any questions or contributions, please open an issue or submit a pull request.*
