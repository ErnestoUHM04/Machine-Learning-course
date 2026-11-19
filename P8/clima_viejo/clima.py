# Hernández Martínez Ernesto Ulises
# Practica 8: Predicción del clima en la Ciudad de México

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split


# BUSCAR MODELOS
# LinealReegression
from sklearn.linear_model import LinearRegression
# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# MLPRegressor
from sklearn.neural_network import MLPRegressor
# HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
# GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# Regresion no Parametrica


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Para descargar los datos del clima de la CDMX se usó el siguiente link
# https://dev.meteostat.net/bulk/hourly.html#endpoints
# En este se explica cómo bajar los datos necesarios

# Datos por hora de 'Mexico City Airport'       años 2020 - 2024
# https://data.meteostat.net/hourly/2025/76679.csv.gz
# https://data.meteostat.net/hourly/2024/76679.csv.gz
# https://data.meteostat.net/hourly/2023/76679.csv.gz
# https://data.meteostat.net/hourly/2022/76679.csv.gz
# https://data.meteostat.net/hourly/2021/76679.csv.gz
# https://data.meteostat.net/hourly/2020/76679.csv.gz

# Datos por hora de 'Mexico Central, D. F.'     años 2020 - 2024
# https://data.meteostat.net/hourly/2024/76680.csv.gz
# https://data.meteostat.net/hourly/2023/76680.csv.gz
# https://data.meteostat.net/hourly/2022/76680.csv.gz
# https://data.meteostat.net/hourly/2021/76680.csv.gz
# https://data.meteostat.net/hourly/2020/76680.csv.gz

"""
clima_76680_2020 = pd.read_csv('76680_2020.csv')
clima_76679_2020 = pd.read_csv('76679_2020.csv')
clima_76680_2021 = pd.read_csv('76680_2021.csv')
clima_76679_2021 = pd.read_csv('76679_2021.csv')
clima_76680_2022 = pd.read_csv('76680_2022.csv')
clima_76679_2022 = pd.read_csv('76679_2022.csv')
clima_76680_2023 = pd.read_csv('76680_2023.csv')
clima_76679_2023 = pd.read_csv('76679_2023.csv')
clima_76680_2024 = pd.read_csv('76680_2024.csv')
clima_76679_2024 = pd.read_csv('76679_2024.csv')

clima_cdmx = pd.concat([clima_76680_2020, clima_76679_2020, clima_76680_2021, clima_76679_2021, clima_76680_2022, clima_76679_2022, clima_76680_2023, clima_76679_2023, clima_76680_2024, clima_76679_2024])


# Selección de columnas relevantes
clima_cdmx = clima_cdmx[['year', 'month', 'day', 'hour', 'temp']] # is the year really needed?

clima_cdmx.to_csv('clima_cdmx_completo.csv', index=False)
print("Dataset combinado guardado como clima_cdmx_completo.csv")
"""
# Cargar el dataset combinado
clima_cdmx = pd.read_csv('clima_cdmx_completo.csv')

# PARA GAUSSIAN PROCESS REGRESSOR hay que hacer un sampling para reducir el dataset
#clima_cdmx = clima_cdmx.sample(8000, random_state=42)

print(clima_cdmx.head())


X = clima_cdmx[['month', 'day', 'hour']]
y = clima_cdmx['temp']

# Dividir 80 - 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo de regresión lineal
#linear_regression_model = LinearRegression()

#randomforest_regressor_model = RandomForestRegressor(n_estimators=200, max_depth=1000, max_features=3, random_state=66)

mlp_regressor_model = MLPRegressor(hidden_layer_sizes=(200,200,200), activation='relu', solver='adam', max_iter=500, random_state=66)

#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#kernel = 1.0 * RBF(length_scale=5.0)

#gaussianprocess_regressor_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, random_state=66)

#linear_regression_model.fit(X_train, y_train)

#gaussianprocess_regressor_model.fit(X_train, y_train)

mlp_regressor_model.fit(X_train, y_train)
#randomforest_regressor_model.fit(X_train, y_train)

# Evaluación del modelo
#score = linear_regression_model.score(X_test, y_test)
#print(f'Precisión del modelo de regresión lineal: {score * 100:.2f}%')

# Métricas de evaluación
#y_pred = linear_regression_model.predict(X_test)

# Predicciones
#y_pred = randomforest_regressor_model.predict(X_test)
y_pred = mlp_regressor_model.predict(X_test)

#y_pred, sigma = gaussianprocess_regressor_model.predict(X_test, return_std=True)

# Cálculo de métricas
mea = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mea:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

""""
# --- Gráfica ligera con muestra aleatoria (scatter 3D) ---
sample = clima_cdmx.sample(3000, random_state=42)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(sample['month'], sample['day'], sample['temp'], s=5, alpha=0.4, label='Real (muestra)', color='blue')
#ax.scatter(sample['month'], sample['day'], linear_regression_model.predict(sample[['month','day','hour']]), s=5, alpha=0.4, label='Predicho (muestra)', color='red')
#ax.scatter(sample['month'], sample['day'], randomforest_regressor_model.predict(sample[['month','day','hour']]), s=5, alpha=0.4, label='Predicho (muestra)', color='red')
#ax.scatter(sample['month'], sample['day'], mlp_regressor_model.predict(sample[['month','day','hour']]), s=5, alpha=0.4, label='Predicho (muestra)', color='green')
ax.scatter(sample['month'], sample['day'], gaussianprocess_regressor_model.predict(sample[['month','day','hour']]), s=5, alpha=0.4, label='Predicho (muestra)', color='red')

ax.set_xlabel('Mes')
ax.set_ylabel('Día')
ax.set_zlabel('Temperatura (°C)')
ax.set_title('Temperaturas Reales vs Predichas (Muestra 3D)')
ax.legend()

plt.show()

# --- Superficie promediada por mes y día (3D Surface) ---
prom = clima_cdmx.groupby(['month','day'])['temp'].mean().reset_index()
Xg = prom['month'].values
Yg = prom['day'].values
Zg = prom['temp'].values

grid_x, grid_y = np.mgrid[1:12:50j, 1:31:50j]
grid_z = griddata((Xg, Yg), Zg, (grid_x, grid_y), method='cubic')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.7, cmap='viridis')
ax.set_xlabel('Mes')
ax.set_ylabel('Día')
ax.set_zlabel('Temperatura promedio (°C)')
ax.set_title('Superficie promedio de temperatura (Mes vs Día)')

plt.show()


# lets save the model
#joblib.dump(randomforest_regressor_model, 'randomforest_clima_cdmx_model.pkl')
#print("Modelo guardado como randomforest_clima_cdmx_model.pkl")

#joblib.dump(mlp_regressor_model, 'mlp_clima_cdmx_model.pkl')
#print("Modelo guardado como mlp_clima_cdmx_model.pkl")

joblib.dump(gaussianprocess_regressor_model, 'gaussianprocess_clima_cdmx_model.pkl')
print("Modelo guardado como gaussianprocess_clima_cdmx_model.pkl")
"""