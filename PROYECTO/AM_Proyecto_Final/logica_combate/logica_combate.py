import joblib
import pandas as pd
import numpy as np
import warnings

# Suprimir warnings de sklearn sobre nombres de características
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class SistemaCombate:
    def __init__(
        self, model_path="modelo_rf.pkl", dataset_path="dataset_peces_enemigos.csv"
    ):
        # Cargar el modelo entrenado
        self.modelo = joblib.load(model_path)
        # Cargar el dataset original para poder "spawnear" peces reales
        self.df_peces = pd.read_csv(dataset_path)
        self.columnas = [
            "Nivel",
            "Vida",
            "Fuerza",
            "Resistencia",
            "Agresividad",
            "Amenaza",
            "Recompensa",
            "Experiencia",
        ]

    def generar_enemigo_por_zona(self, zona_actual):
        """Filtra y devuelve un pez aleatorio según la amenaza permitida por zona"""
        # Definición de reglas de amenaza por zona
        if zona_actual in [0, 1, 2, 3]:  # Zonas 1-4 (índice 0-3)
            amenazas_permitidas = [1, 2]
        elif zona_actual in [4, 5, 6, 7]:  # Zonas 5-8 (índice 4-7)
            amenazas_permitidas = [3, 4]
        else:  # Zonas 9-12 (índice 8-11)
            amenazas_permitidas = [5, 6]

        # Filtrar peces que cumplen la condición de amenaza
        opciones = self.df_peces[self.df_peces["Amenaza"].isin(amenazas_permitidas)]

        # Seleccionar uno al azar y devolverlo como diccionario
        pez_data = opciones.sample(n=1).iloc[0].to_dict()
        return pez_data

    def procesar_encuentro(self, pez_stats, rod_lvl):
        """
        Usa el modelo para identificar al pez y determinar el resultado.
        Recibe las stats del pez y el nivel de la caña del agente.
        """
        # Preparar datos para el modelo (mismo orden que el entrenamiento)
        datos_fila = [
            [
                pez_stats["Nivel"],
                pez_stats["Vida"],
                pez_stats["Fuerza"],
                pez_stats["Resistencia"],
                pez_stats["Agresividad"],
                pez_stats["Amenaza"],
                pez_stats["Recompensa"],
                pez_stats["Experiencia"],
            ]
        ]

        X = pd.DataFrame(datos_fila, columns=self.columnas)

        # Identificación via Modelo
        identificacion = self.modelo.predict(X)[0]

        # Lógica de victoria: Basada en la fuerza del pez vs nivel de la caña (rod_lvl)
        # Mejorada para dar más oportunidad a niveles bajos
        probabilidad_victoria = min(1.0, (rod_lvl * 12) / pez_stats["Fuerza"])
        gana_agente = np.random.random() < probabilidad_victoria

        if gana_agente:
            return {
                "resultado": "victoria",
                "identificacion": identificacion,
                "recompensa": pez_stats["Recompensa"],
                "experiencia": pez_stats["Experiencia"],
                "amenaza": pez_stats["Amenaza"],
            }
        else:
            # Penalización: Restamos proporcional a la amenaza (balanceado)
            penalizacion = int(pez_stats["Amenaza"] * 2)  # Reducido de *10 a *2
            return {
                "resultado": "derrota",
                "identificacion": identificacion,
                "recompensa": -penalizacion,
            }
