import joblib
import pandas as pd
import numpy as np

import datetime
from meteostat import Hourly, Stations

model = joblib.load('models/hgb_clima_cdmx_more_features_model.pkl')

print("\nPredicción de temperatura en la Ciudad de México\n")

"""
while True:
    year = int(input("Ingrese el año (2020-2025): "))
    if 2020 <= year <= 2025:
        break
    else:
        print("Año inválido. Por favor ingrese un valor entre 2020 y 2025.")
"""

while True:
    month = int(input("Ingrese el mes (1-12): "))
    if 1 <= month <= 12:
        break
    else:
        print("Mes inválido. Por favor ingrese un valor entre 1 y 12.")

while True:
    if month == 2:
        day = int(input("Ingrese el día (1-29): "))
        if 1 <= day <= 29:
            break
        else:
            print("Día inválido para febrero. Por favor ingrese un valor entre 1 y 29.")
    elif month in [4, 6, 9, 11]:
        day = int(input("Ingrese el día (1-30): "))
        if 1 <= day <= 30:
            break
        else:
            print("Día inválido. Por favor ingrese un valor entre 1 y 30.")
    else:
        day = int(input("Ingrese el día (1-31): "))
        if 1 <= day <= 31:
            break
        else:
            print("Día inválido. Por favor ingrese un valor entre 1 y 31.")

while True:
    hour = int(input("Ingrese la hora (0-23): "))
    if 0 <= hour <= 23:
        break
    else:
        print("Hora inválida. Por favor ingrese un valor entre 0 y 23.")

# We have to compute the additional features used in the more features model

# FEATURES -  hour_sin, hour_cos, doy_sin, doy_cos

# We are mising temp_prev and temp_roll6, we will set them to average values
# These values can be adjusted based on historical data if available
#avg_temp_prev = 0.0  # Example average previous temperature
#avg_temp_roll6 = 0.0  # Example average rolling temperature


end = datetime.datetime(2025, month, day, hour)
start = end - datetime.timedelta(hours=6)
data = Hourly("76680", start, end).fetch()

if not data.empty:
    temps = data['temp'].dropna().to_list()
    print(temps)
    temp_prev = temps[-1] if len(temps) >= 1 else 20.0
    print(temp_prev)
    temp_roll6 = np.mean(temps[-6:]) if len(temps) >= 6 else temp_prev
    print(temp_roll6)
else:
    print("No se pudieron obtener datos históricos, usando valores por defecto")
    temp_prev = 20.0
    temp_roll6 = 20.0

"""
while True:
    try:
        avg_temp_prev = float(input("Ingrese la temperatura promedio de la hora previa (°C): "))
        break
    except ValueError:
        print("Entrada inválida. Por favor ingrese un número válido para la temperatura.")

avg_temp_roll6 = avg_temp_prev  # For simplicity, we use the same value
"""

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

day_of_year = (month - 1) * 30 + day

doy_sin = np.sin(2 * np.pi * day_of_year / 365)
doy_cos = np.cos(2 * np.pi * day_of_year / 365)

#input_data = pd.DataFrame([[hour_sin, hour_cos, doy_sin, doy_cos]], columns=['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos'])
input_data = pd.DataFrame([[hour_sin, hour_cos, doy_sin, doy_cos, temp_prev, temp_roll6]], columns=['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'temp_prev', 'temp_roll6'])

prediccion_temp = model.predict(input_data)

print(f'La temperatura predicha para el {day}/{month} a las {hour}:00 es de {prediccion_temp[0]:.2f} °C')