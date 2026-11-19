import pandas as pd
import numpy as np
from meteostat import Hourly, Point
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

import joblib

# Coordinates for Mexico City center
cdmx = Point(19.4326, -99.1332)

# Define date range
start = datetime(2020, 1, 1)
end = datetime(2024, 12, 31)

# Fetch data for station 76680 (CDMX)
data = Hourly(cdmx, start, end, model=True)
data = data.fetch()

# Keep only needed columns
data = data[["temp"]].reset_index()
data.rename(columns={"time": "datetime"}, inplace=True)

# Drop missing values
data = data.dropna()
print(data.head(24))

data["year"] = data["datetime"].dt.year
data["month"] = data["datetime"].dt.month
data["day"] = data["datetime"].dt.day
data["hour"] = data["datetime"].dt.hour
data["day_of_year"] = data["datetime"].dt.dayofyear

# Cyclical time encodings
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data["doy_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365)
data["doy_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365)

data["temp_prev"] = data["temp"].shift(1)
data["temp_roll6"] = data["temp"].rolling(window=6).mean()

# Drop first few rows with NaNs
data = data.dropna().reset_index(drop=True)

# Remove obviously unrealistic data
data = data[(data["temp"] >= -5) & (data["temp"] <= 40)]

# Additional night filter: if hour in [0, 6] and temp > 22, mark as invalid
data = data[~((data["hour"].between(0, 6)) & (data["temp"] > 22))]

data.to_csv("cdmx_hourly_clean_2020_2024.csv", index=False)
print("✅ Dataset saved with", len(data), "rows")


# Define features
features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "temp_prev", "temp_roll6"]
X = data[features]
y = data["temp"]

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model = HistGradientBoostingRegressor(max_depth=10, learning_rate=0.05, max_bins=255, random_state=42)
model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, random_state=42)


model.fit(X_train, y_train)

print("R²:", model.score(X_test, y_test))

joblib.dump(model, "models/hgb_clima_corrected_cdmx_2020_2024.pkl")

