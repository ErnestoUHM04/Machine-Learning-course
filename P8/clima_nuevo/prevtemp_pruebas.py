import joblib
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import requests

model = joblib.load('models/hgb_clima_open_meteo_cdmx_2000_2024.pkl')

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

# We need to set temp_prev and temp_roll6
# === Fetch last 6 hours of real data from Open-Meteo ===
def get_recent_temps(target_time, hours_back=6, max_back=24):
    """Fetch up to `hours_back` hours before `target_time`, going further if needed."""
    start_date = (target_time - timedelta(hours=max_back)).strftime("%Y-%m-%d")
    end_date = target_time.strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 19.4326,
        "longitude": -99.1332,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "America/Mexico_City"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = pd.DataFrame(r.json()["hourly"])
    data["time"] = pd.to_datetime(data["time"])
    data.rename(columns={"temperature_2m": "temp"}, inplace=True)
    data = data[["time", "temp"]].dropna()

    # Keep only rows strictly before target_time
    data = data[data["time"] < target_time]

    # Get the most recent 6 valid readings
    temps = data.tail(hours_back)["temp"].to_list()

    # If we don't have enough, try expanding window
    back_expand = hours_back
    while len(temps) < hours_back and back_expand < max_back:
        back_expand += 3
        temps = data.tail(back_expand)["temp"].to_list()

    return temps[-hours_back:] if len(temps) >= hours_back else temps


# === Fetch nearby valid temperatures ===
target_time = datetime(datetime.now().year, month, day, hour)
temps = get_recent_temps(target_time)
if not temps:
    raise RuntimeError("No se encontraron temperaturas válidas cercanas al momento indicado.")

temp_prev = temps[-1]
temp_roll6 = np.mean(temps)

print("\nÚltimas temperaturas usadas:", temps)
print(f"temp_prev = {temp_prev:.2f} °C | temp_roll6 = {temp_roll6:.2f} °C\n")


input_data = pd.DataFrame([[hour_sin, hour_cos, doy_sin, doy_cos, temp_prev, temp_roll6]], columns=['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'temp_prev', 'temp_roll6'])

prediccion_temp = model.predict(input_data)

print(f'La temperatura predicha para el día: {day}/ mes: {month} a las {hour}:00 es de {prediccion_temp[0]:.2f} °C')