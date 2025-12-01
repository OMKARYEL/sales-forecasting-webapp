import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

st.title("📈 Advanced Sales Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload sales.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())
    
    # Clean
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').asfreq('MS')
    st.write("### Processed Time-Series", df)

    # Decompose
    st.write("### Trend / Seasonality / Residuals")
    decomp = seasonal_decompose(df['sales'], model='additive', period=12)
    fig = decomp.plot()
    st.pyplot(fig)

    # Auto ARIMA
    st.write("### Auto ARIMA Model Selection")
    model = auto_arima(df['sales'], seasonal=True, m=12, trace=True)
    st.code(model.summary())

    # Fit SARIMA
    final_model = SARIMAX(df['sales'], order=model.order, seasonal_order=model.seasonal_order)
    result = final_model.fit()

    # Forecast
    st.write("### Forecast Next Months")
    periods = st.slider("Months to Forecast", 6, 36, 12)
    forecast = result.get_forecast(periods)
    pred = forecast.predicted_mean
    conf = forecast.conf_int()

    # Plot forecast
    fig2, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['sales'], label='Historical')
    ax.plot(pred, label='Forecast')
    ax.fill_between(conf.index, conf.iloc[:,0], conf.iloc[:,1], alpha=0.2)
    ax.legend()
    st.pyplot(fig2)

    st.write("### Forecast Table")
    st.dataframe(pred)

else:
    st.info("Upload a CSV file with 'date' and 'sales' columns to begin.")
