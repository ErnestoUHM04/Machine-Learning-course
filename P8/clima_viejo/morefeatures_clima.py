import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

clima_cdmx = pd.read_csv('clima_cdmx_completo.csv')

# Add day of year
clima_cdmx['day_of_year'] = (clima_cdmx['month'] - 1) * 30 + clima_cdmx['day']

# Cyclical encoding for hour and day of year
clima_cdmx['hour_sin'] = np.sin(2 * np.pi * clima_cdmx['hour'] / 24)
clima_cdmx['hour_cos'] = np.cos(2 * np.pi * clima_cdmx['hour'] / 24)

clima_cdmx['doy_sin'] = np.sin(2 * np.pi * clima_cdmx['day_of_year'] / 365)
clima_cdmx['doy_cos'] = np.cos(2 * np.pi * clima_cdmx['day_of_year'] / 365)

# Add station ID based on source file.  <- This probably works but i need to verify it
#clima_cdmx['station'] = clima_cdmx['source'].astype('category').cat.codes  # you need to add this column earlier if not present

# Lag feature: previous hour temp
clima_cdmx['temp_prev'] = clima_cdmx['temp'].shift(1)

# Rolling mean (previous 6 hours)
clima_cdmx['temp_roll6'] = clima_cdmx['temp'].rolling(6).mean()

# Drop missing from lag/rolling
clima_cdmx = clima_cdmx.dropna()

features = [
    'hour_sin','hour_cos',
    'doy_sin','doy_cos',
    'temp_prev','temp_roll6'
]

clima_cdmx.to_csv('clima_cdmx_more_features.csv', index=False)
print("Dataset combinado guardado como clima_cdmx_more_features.csv")

X = clima_cdmx[features]
y = clima_cdmx['temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(
    max_depth=8,
    learning_rate=0.05,
    max_bins=255,
    random_state=66
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

joblib.dump(model, 'hgb_clima_cdmx_more_features_model.pkl')
print("Modelo guardado como hgb_clima_cdmx_more_features_model.pkl")
