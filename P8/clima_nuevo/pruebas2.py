import joblib
import pandas as pd
import numpy as np

model = joblib.load('models/hgb_clima_woprev_open_meteo_cdmx_2000_2024.pkl')

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

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

day_of_year = (month - 1) * 30 + day

doy_sin = np.sin(2 * np.pi * day_of_year / 365)
doy_cos = np.cos(2 * np.pi * day_of_year / 365)

input_data = pd.DataFrame([[hour_sin, hour_cos, doy_sin, doy_cos]], columns=['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos'])

prediccion_temp = model.predict(input_data)

print(f'La temperatura predicha para el día: {day}/ mes: {month} a las {hour}:00 es de {prediccion_temp[0]:.2f} °C')