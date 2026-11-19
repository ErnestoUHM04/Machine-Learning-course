"""
Script de debug para probar la generación y predicción de objetos lógicos
"""

from logica_objetos import ObjetoLogico, RAREZA_COMUN, RAREZA_LEGENDARIO

# Configuración para cada tipo de objeto (basada en el dataset de entrenamiento)
objetos_config = {
    "cana_basica": {
        "rareza": (1, 2),
        "nivel_requerido": (1, 15),
        "precio_venta": 40,
        "porcentaje_completado": (0, 1),
        "valor_estrategico": (1, 2),
    },
    "cana_rara": {
        "rareza": (2, 3),
        "nivel_requerido": (10, 40),
        "precio_venta": 160,
        "porcentaje_completado": (1, 2),
        "valor_estrategico": (2, 4),
    },
    "cana_epica": {
        "rareza": (3, 4),
        "nivel_requerido": (30, 60),
        "precio_venta": 640,
        "porcentaje_completado": (2, 3),
        "valor_estrategico": (4, 8),
    },
    "cana_legendaria": {
        "rareza": (4, 5),
        "nivel_requerido": (50, 100),
        "precio_venta": 2560,
        "porcentaje_completado": (3, 3),  # Valor fijo como tupla
        "valor_estrategico": (8, 8),
    },
    "anillas_sixpack": {
        "rareza": (1, 2),
        "nivel_requerido": (1, 100),
        "precio_venta": 0,
        "porcentaje_completado": (0, 1),
        "valor_estrategico": (-1, 0),
    },
    "botella": {
        "rareza": (2, 3),
        "nivel_requerido": (1, 100),
        "precio_venta": 0,
        "porcentaje_completado": (0, 1),
        "valor_estrategico": (-2, -1),
    },
    "lata": {
        "rareza": (3, 4),
        "nivel_requerido": (1, 100),
        "precio_venta": 0,
        "porcentaje_completado": (0, 1),
        "valor_estrategico": (-3, -2),
    },
    "llanta": {
        "rareza": (4, 5),
        "nivel_requerido": (1, 100),
        "precio_venta": 0,
        "porcentaje_completado": (0, 1),
        "valor_estrategico": (-4, -3),
    },
    "tesoro": {
        "rareza": (5, 5),
        "nivel_requerido": (1, 100),
        "precio_venta": 1280,
        "porcentaje_completado": (2, 3),
        "valor_estrategico": (10, 10),
    },
}

print("=" * 80)
print("DEBUG: Generación y Predicción de Objetos Lógicos")
print("=" * 80)
print()

# Generar varios objetos de cada tipo para ver la distribución
num_objetos_por_tipo = 3
objetos = []

print(f"Generando {num_objetos_por_tipo} objetos de cada tipo...\n")

for tipo_esperado, config in objetos_config.items():
    print(f"--- Tipo esperado: {tipo_esperado} ---")
    for i in range(num_objetos_por_tipo):
        objeto = ObjetoLogico(
            config, usar_rareza_ponderada=False, predecir_automatico=True
        )
        objetos.append((tipo_esperado, objeto))

        print(f"  Objeto #{i+1}:")
        print(f"    Rareza: {objeto.rareza} ({objeto.get_nombre_rareza()})")
        print(f"    Nivel requerido: {objeto.nivel_requerido}")
        print(f"    Porcentaje completado: {objeto.porcentaje_completado}")
        print(f"    Valor estratégico: {objeto.valor_estrategico}")
        print(f"    --> PREDICCIÓN: {objeto.tipo_objeto}")
        print(
            f"    --> Correcto: {'✓' if objeto.tipo_objeto == tipo_esperado else '✗'}"
        )
    print()

# Estadísticas de precisión
print("=" * 80)
print("ESTADÍSTICAS DE PREDICCIÓN")
print("=" * 80)

correctos = sum(1 for tipo_esperado, obj in objetos if obj.tipo_objeto == tipo_esperado)
total = len(objetos)
precision = (correctos / total) * 100

print(f"\nPrecisión general: {correctos}/{total} ({precision:.1f}%)")

# Matriz de confusión simplificada
print("\nPredicciones por tipo:")
conteo_por_tipo = {}
for tipo_esperado, obj in objetos:
    key = (tipo_esperado, obj.tipo_objeto)
    conteo_por_tipo[key] = conteo_por_tipo.get(key, 0) + 1

for (esperado, predicho), cantidad in sorted(conteo_por_tipo.items()):
    simbolo = "✓" if esperado == predicho else "✗"
    print(
        f"  {simbolo} Esperado: {esperado:20s} -> Predicho: {predicho:20s} ({cantidad})"
    )

print("\n✓ Debug completado exitosamente")
