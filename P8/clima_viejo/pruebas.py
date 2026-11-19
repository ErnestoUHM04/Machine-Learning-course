
import joblib
import pandas as pd

# Cargar el modelo guardado
#model = joblib.load('randomforest_clima_cdmx_model.pkl')
#model = joblib.load('mlp_clima_cdmx_model.pkl')
#model = joblib.load('gaussianprocess_clima_cdmx_model.pkl')
model = joblib.load('models/hgb_clima_cdmx_model.pkl')
#model = joblib.load('hgb_clima_cdmx_more_features_model.pkl')

# Input del usuario para predecir la temperatura
# Ejemplo: mes=5, día=15, hora=14
# formato 24 horas
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


input_data = pd.DataFrame([[month, day, hour]], columns=['month', 'day', 'hour'])
#prediccion_temp = linear_regression_model.predict(input_data)

prediccion_temp = model.predict(input_data)

print(f'La temperatura predicha para el {day}/{month} a las {hour}:00 es de {prediccion_temp[0]:.2f} °C')

# Para GPR, mostrar también la desviación estándar
#if hasattr(model, 'predict'):
#    prediccion_temp, desviacion_estandar = model.predict(input_data, return_std=True)
#    print(f'La temperatura predicha para el {day}/{month} a las {hour}:00 es de {prediccion_temp[0]:.2f} °C')
#    print(f'Desviación estándar: {desviacion_estandar[0]:.2f} °C')