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

## Deploy on Streamlit Cloud
1. Push these files to a **public GitHub repo**.
2. Go to https://share.streamlit.io, click **Deploy an app**.
3. Select your repo, branch (main), file path = `app.py`.
4. Deploy → you’ll get a public app URL.
