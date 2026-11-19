import random
import pickle
import os

"""
Toma el minimo y maximo de rareza, nivel requerido, precio de venta, porcentaje completado y valor estrategico 
para generar valores aleatorios con clases sin nombre para usar el modelo y que prediga el objeto logico.

Asi como una tabla del precio de venta segun el nombre para que el agente pueda venderlo y acumular oro/dinero
Para pasar de mapa a mapa. (lo de pasar al mapa no es necesario aun).
"""

# Cargar el modelo una sola vez al importar el módulo
_directorio_actual = os.path.dirname(os.path.abspath(__file__))
_ruta_modelo = os.path.join(_directorio_actual, "modelo_arbol_decision.pkl")
_modelo = None


def cargar_modelo():
    """Carga el modelo de predicción de objetos"""
    global _modelo
    if _modelo is None:
        if os.path.exists(_ruta_modelo):
            with open(_ruta_modelo, "rb") as f:
                _modelo = pickle.load(f)
            print(f"Modelo cargado desde: {_ruta_modelo}")
        else:
            print(f"Advertencia: No se encontró el modelo en {_ruta_modelo}")
    return _modelo


# Tabla de precios de venta según el nombre del objeto lógico
tabla_precios_venta = {
    "cana_basica": 40,
    "cana_rara": 160,
    "cana_epica": 640,
    "cana_legendaria": 2560,
    "anillas_sixpack": 0,
    "botella": 0,
    "lata": 0,
    "llanta": 0,
    "tesoro": 1280,
}

# Niveles de rareza
RAREZA_COMUN = 1
RAREZA_RARO = 2
RAREZA_ULTRA_RARO = 3
RAREZA_EPICO = 4
RAREZA_LEGENDARIO = 5

# Probabilidades para cada nivel de rareza (suma = 100)
PROBABILIDADES_RAREZA = {
    RAREZA_COMUN: 50,  # 50% común
    RAREZA_RARO: 30,  # 30% raro
    RAREZA_ULTRA_RARO: 13,  # 13% ultra raro
    RAREZA_EPICO: 6,  # 6% épico
    RAREZA_LEGENDARIO: 1,  # 1% legendario (muy difícil)
}


def generar_rareza_ponderada():
    """Genera una rareza basada en probabilidades ponderadas"""
    valores = list(PROBABILIDADES_RAREZA.keys())
    pesos = list(PROBABILIDADES_RAREZA.values())
    return random.choices(valores, weights=pesos, k=1)[0]


class ObjetoLogico:
    def __init__(self, config, usar_rareza_ponderada=True, predecir_automatico=True):
        # Guardar todo en un diccionario para fácil manipulación
        self.datos = {
            "tipo_objeto": None,  # Se asignará después de la predicción
            "rareza": (
                generar_rareza_ponderada()
                if usar_rareza_ponderada
                else random.randint(config["rareza"][0], config["rareza"][1])
            ),
            "nivel_requerido": random.randint(
                config["nivel_requerido"][0], config["nivel_requerido"][1]
            ),
            "precio_venta": config["precio_venta"],
            "porcentaje_completado": random.randint(
                config["porcentaje_completado"][0],
                config["porcentaje_completado"][1],
            ),
            "valor_estrategico": random.randint(
                config["valor_estrategico"][0], config["valor_estrategico"][1]
            ),
        }

        # Predecir automáticamente el tipo de objeto si está habilitado
        if predecir_automatico:
            self.predecir_tipo()

    # Propiedades para mantener compatibilidad con código existente
    @property
    def rareza(self):
        return self.datos["rareza"]

    @property
    def nivel_requerido(self):
        return self.datos["nivel_requerido"]

    @property
    def precio_venta(self):
        return self.datos["precio_venta"]

    @property
    def porcentaje_completado(self):
        return self.datos["porcentaje_completado"]

    @property
    def valor_estrategico(self):
        return self.datos["valor_estrategico"]

    @property
    def tipo_objeto(self):
        return self.datos["tipo_objeto"]

    def get_nombre_rareza(self):
        """Retorna el nombre legible de la rareza"""
        nombres_rareza = {
            RAREZA_COMUN: "Común",
            RAREZA_RARO: "Raro",
            RAREZA_ULTRA_RARO: "Ultra Raro",
            RAREZA_EPICO: "Épico",
            RAREZA_LEGENDARIO: "Legendario",
        }
        return nombres_rareza.get(self.datos["rareza"], "Desconocido")

    def get_vector(self):
        """Retorna el vector de características para el modelo ML"""
        return [
            self.datos["rareza"],
            self.datos["nivel_requerido"],
            self.datos["precio_venta"],
            self.datos["porcentaje_completado"],
            self.datos["valor_estrategico"],
        ]

    def predecir_tipo(self):
        """Usa el modelo para predecir el tipo de objeto y actualiza el diccionario"""
        modelo = cargar_modelo()
        if modelo is not None:
            # Preparar datos en el formato: [[rareza, nivel, precio, porcentaje, valor]]
            rareza = self.datos["rareza"]
            nivel = self.datos["nivel_requerido"]
            precio = self.datos["precio_venta"]
            porcentaje = self.datos["porcentaje_completado"]
            valor = self.datos["valor_estrategico"]

            prediccion = modelo.predict([[rareza, nivel, precio, porcentaje, valor]])
            tipo_predicho = prediccion[0]
            self.set_tipo_objeto(tipo_predicho)
            return tipo_predicho
        else:
            print("No se pudo predecir: modelo no disponible")
            return None

    def set_tipo_objeto(self, tipo):
        """Asigna el tipo de objeto predicho por el modelo"""
        self.datos["tipo_objeto"] = tipo
        # Actualizar precio_venta según la tabla
        if tipo in tabla_precios_venta:
            self.datos["precio_venta"] = tabla_precios_venta[tipo]

    def __repr__(self):
        return (
            f"tipo_objeto={self.datos['tipo_objeto']}, "
            f"rareza={self.datos['rareza']}, "
            f"nivel_requerido={self.datos['nivel_requerido']}, "
            f"precio_venta={self.datos['precio_venta']}, "
            f"porcentaje_completado={self.datos['porcentaje_completado']}, "
            f"valor_estrategico={self.datos['valor_estrategico']})"
        )
