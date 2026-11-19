import numpy as np
import requests
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 19.4326,
    "longitude": -99.1332,
    "start_date": "2000-01-01",
    "end_date": "2024-12-31",
    "hourly": "temperature_2m",
    "timezone": "America/Mexico_City"
}

response = requests.get(url, params=params)
data = pd.DataFrame(response.json()["hourly"])
data["time"] = pd.to_datetime(data["time"])
data.rename(columns={"time": "datetime", "temperature_2m": "temp"}, inplace=True)
print(data.head(24))

data["year"] = data["datetime"].dt.year
data["month"] = data["datetime"].dt.month
data["day"] = data["datetime"].dt.day
data["hour"] = data["datetime"].dt.hour
data["day_of_year"] = data["datetime"].dt.dayofyear

data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data["doy_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365)
data["doy_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365)

data["temp_prev"] = data["temp"].shift(1)
data["temp_roll6"] = data["temp"].rolling(6).mean()

data = data.dropna()

data.to_csv("cdmx_hourly_open_meteo_2000_2024.csv", index=False)
print("✅ Dataset saved with", len(data), "rows")


# Define features
features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "temp_prev", "temp_roll6"]
#features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
X = data[features]
y = data["temp"]

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(
    max_depth=12,
    learning_rate=0.03,
    max_iter=800,
    min_samples_leaf=10,
    l2_regularization=0.2,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=66
)

#model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Metrics
y_pred = model.predict(X_test)

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

joblib.dump(model, "models/hgb_clima_open_meteo_cdmx_2000_2024.pkl")
