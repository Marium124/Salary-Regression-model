import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
import pickle

st.set_page_config(page_title="Salary Regression ANN", page_icon="💼", layout="wide")
st.title("💼 Salary Regression — ANN with Gender Encoder")

@st.cache_data
def load_data():
    return pd.read_csv("sample_data.csv")

@st.cache_resource
def load_gender_encoder():
    with open("label_encoder_gender.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
label_encoder_gender = load_gender_encoder()

st.subheader("📄 Data Preview")
st.dataframe(df.head(15), use_container_width=True)
st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ ANN Settings")
    n_layers = st.slider("Hidden layers", 1, 5, 3)
    units = st.slider("Units per layer", 8, 256, 64, step=8)
    dropout = st.slider("Dropout", 0.0, 0.6, 0.1, step=0.05)
    learning_rate = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
    epochs = st.slider("Epochs", 50, 1000, 200, step=50)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=64)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", value=42, step=1)
    train_btn = st.button("🚀 Train Model", use_container_width=True)

# Apply gender encoding with provided pkl
if "Gender" not in df.columns:
    df["Gender"] = np.random.choice(["Male","Female"], size=len(df))

df["Gender"] = label_encoder_gender.transform(df["Gender"].astype(str))

target_col = "Salary"
y = df[target_col]
X = df.drop(columns=[target_col])

# Identify types
cat_cols = [c for c in X.select_dtypes(include=["object","category"]).columns if c!="Gender"]
num_cols = X.select_dtypes(include=[np.number,"float","int"]).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer([("num", numeric_pipeline, num_cols),
                                  ("cat", categorical_pipeline, cat_cols)], remainder="drop")

def build_model(input_dim, units, n_layers, dropout, lr):
    model = keras.Sequential(name="salary_regressor")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for i in range(n_layers):
        model.add(keras.layers.Dense(units, activation="relu"))
        if dropout>0: model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="linear"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model

if train_btn:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = build_model(X_train_t.shape[1], units, n_layers, dropout, learning_rate)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    hist = model.fit(X_train_t, y_train, validation_split=0.2, epochs=epochs,
                     batch_size=batch_size, verbose=0, callbacks=[es])

    preds = model.predict(X_test_t).ravel()
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(np.mean((y_test - preds)**2))
    r2 = r2_score(y_test, preds)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("R²", f"{r2:,.4f}")

    # Plot loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=hist.history["loss"], mode="lines", name="Train Loss"))
    fig_loss.add_trace(go.Scatter(y=hist.history["val_loss"], mode="lines", name="Val Loss"))
    fig_loss.update_layout(title="Loss curves", xaxis_title="Epoch", yaxis_title="MSE")
    st.plotly_chart(fig_loss, use_container_width=True)

    # Pred vs Actual
    fig_sc = px.scatter(x=y_test, y=preds, labels={"x":"Actual","y":"Predicted"}, title="Predicted vs Actual")
    fig_sc.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines", name="Ideal"))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Prediction form
    st.subheader("🧪 Try a Prediction")
    with st.form("predict_form"):
        row = {}
        for col in X.columns:
            if col=="Gender":
                options = ["Male","Female"]
                row[col] = st.selectbox(col, options=options)
            elif col in cat_cols:
                options = df[col].dropna().unique().tolist()
                row[col] = st.selectbox(col, options=options)
            else:
                row[col] = st.number_input(col, value=float(X[col].median()))
        submit_pred = st.form_submit_button("Predict")
        if submit_pred:
            row_df = pd.DataFrame([row])
            row_df["Gender"] = label_encoder_gender.transform(row_df["Gender"].astype(str))
            row_t = preprocessor.transform(row_df)
            y_hat = float(model.predict(row_t).ravel()[0])
            st.success(f"Predicted Salary: **{y_hat:,.2f}**")
