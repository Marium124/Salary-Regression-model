# Salary Regression ANN (with Gender Encoder)

This Streamlit app trains an Artificial Neural Network (ANN) to predict **Salary** using tabular data.

## Features
- Preprocessing with `label_encoder_gender.pkl` for Gender
- One-hot encoding for Education & City
- Scaling for numeric features (Age, Experience)
- ANN regression with Keras (configurable layers, units, dropout, learning rate, etc.)
- Metrics: MAE, RMSE, R²
- Training vs Validation loss plot
- Predicted vs Actual scatter plot
- Interactive prediction form

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
