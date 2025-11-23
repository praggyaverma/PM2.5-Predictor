import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# Page Title
# -------------------------------------------------------
st.title("🌫️ Beijing PM2.5 Prediction Dashboard")

st.write("""
This dashboard trains a **K-Nearest Neighbours Regression model** on the 
**Beijing PM2.5 dataset** and visualizes predicted vs actual air quality.
""")

# -------------------------------------------------------
# Upload Dataset
# -------------------------------------------------------
st.header("📂 Upload Any File from Beijing PM2.5 Dataset")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Raw Data (Top 5 Rows)")
    st.dataframe(df.head(5))

    # Preprocessing
    st.header("🧹 Data Preprocessing")

    df = df.dropna()  # simple cleaning
    features = ["DEWP", "TEMP", "PRES", "PM10","SO2", "NO2"]
    X = df[features]
    y = df["PM2.5"]

    st.write("Features used:", features)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    # Train KNN Model
    st.header("🤖 Train KNN Model")

    k_value = st.slider("Number of Neighbors (K)", 1, 20, 5)

    model = KNeighborsRegressor(n_neighbors=k_value)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Model Performance
    st.header("📊 Model Performance")

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("Mean Squared Error", f"{mse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")

    # Plot Actual vs Predicted
    st.subheader("📈 Actual vs Predicted PM2.5")

    fig, ax = plt.subplots()
    ax.scatter(y_test[:200], y_pred[:200]) 
    ax.set_xlabel("Actual PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)

    # Predict on Custom Input
    st.header("🧪 Try Custom Prediction")

    st.write("""Enter Air Quality Metrics and let the model guess the PM2.5 Concentration!""")

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.number_input("Temperature", value=5.0)
        dewp = st.number_input("Dew Point", value=-10.0)
    with c2:
        pres = st.number_input("Pressure", value=1020.0)
        pm10 = st.number_input("PM10", value=4.0)
    with c3:
        so2 = st.number_input("SO2", value=4.0)
        no2 = st.number_input("NO2", value=7.0)

    if st.button("Predict PM2.5"):
        sample = np.array([[dewp, temp, pres, pm10, so2, no2]])
        sample_scaled = scaler.transform(sample)
        result = model.predict(sample_scaled)[0]

        st.success(f"Predicted PM2.5: {result:.2f}")

else:
    st.info("👆 Please upload the Beijing PM2.5 dataset CSV to continue.")
