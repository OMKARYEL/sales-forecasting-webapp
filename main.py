import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("data/sales.csv")

print("Columns found:", df.columns)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.asfreq('MS')   # Month start frequency

print(df.head())

# -----------------------
# VISUALIZE RAW SALES
# -----------------------
plt.figure(figsize=(12,5))
plt.plot(df['sales'])
plt.title("Monthly Sales (2018–2023)")
plt.show()

# -----------------------
# DECOMPOSE TREND / SEASONALITY
# -----------------------
result = seasonal_decompose(df['sales'], model='additive')
result.plot()
plt.show()

# -----------------------
# ADF TEST (Stationarity)
# -----------------------
def adf_test(series):
    result = adfuller(series.dropna())
    print("\nADF Statistic:", result[0])
    print("p-value:", result[1])

adf_test(df['sales'])

# -----------------------
# AUTO ARIMA (Automatic model selection)
# -----------------------
print("\nRunning Auto ARIMA…")
auto_model = auto_arima(df['sales'], seasonal=True, m=12, trace=True)
print(auto_model.summary())

# -----------------------
# TRAIN SARIMA MODEL
# -----------------------
model = SARIMAX(
    df['sales'],
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order
)

result = model.fit()
print(result.summary())

# -----------------------
# FORECAST NEXT 12 MONTHS
# -----------------------
forecast = result.get_forecast(steps=12)
pred = forecast.predicted_mean
conf = forecast.conf_int()

print("\nForecast for next 12 months:")
print(pred)

# -----------------------
# PLOT FORECAST
# -----------------------
plt.figure(figsize=(12,5))
plt.plot(df['sales'], label="Historical")
plt.plot(pred, label="Forecast")
plt.fill_between(conf.index, conf.iloc[:,0], conf.iloc[:,1], alpha=0.3)
plt.legend()
plt.show()
