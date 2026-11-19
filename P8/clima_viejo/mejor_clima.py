import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

clima_cdmx = pd.read_csv('clima_cdmx_completo.csv')

X = clima_cdmx[['month', 'day', 'hour']]
y = clima_cdmx['temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hgb_model = HistGradientBoostingRegressor(
    max_depth=10,
    learning_rate=0.05,
    max_bins=255,
    random_state=66
)

hgb_model.fit(X_train, y_train)
y_pred = hgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

"""
joblib.dump(hgb_model, 'hgb_clima_cdmx_model.pkl')
print("Modelo guardado como hgb_clima_cdmx_model.pkl")
"""