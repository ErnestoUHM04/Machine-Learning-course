import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

DB_FILE = "pokemon.json"


# Cargar db
def cargar_db():
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def guardar_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)


# Construir matriz 

def construir_matriz(db):
    all_attributes = set()
    for p in db:
        all_attributes.update(p["caracteristicas"])
    all_attributes = sorted(list(all_attributes))

    X = []
    y = []
    for p in db:
        row = [1 if attr in p["caracteristicas"] else 0 for attr in all_attributes]
        X.append(row)
        y.append(p["pokemon"])

    return np.array(X), np.array(y), all_attributes


# Recorrer el árbol

def recorrer_arbol(clf, all_attributes, y, node_id=0):
    tree = clf.tree_
    # Hoja
    if tree.feature[node_id] == -2:
        values = tree.value[node_id][0]
        winner = y[np.argmax(values)]
        respuesta = input(f"¿Tu Pokémon es {winner}? (s/n): ").lower()
        if respuesta == "s":
            print("¡Lo adiviné!")
            return True
        else:
            print("Oh, me equivoqué")
            return False

    feature_index = tree.feature[node_id]
    feature_name = all_attributes[feature_index]

    answer = input(f"¿Tu Pokémon es '{feature_name}'? (s/n): ").strip().lower()
    next_node = tree.children_right[node_id] if answer == "s" else tree.children_left[node_id]

    return recorrer_arbol(clf, all_attributes, y, next_node)


# Juego principal
def jugar():
    db = cargar_db()
    if not db:
        print("No hay Pokémon en la base de datos. Agrega al menos uno.")
        return

#Juega y entrena IA
    while True:
        X, y, all_attributes = construir_matriz(db)
        clf = DecisionTreeClassifier()
        clf.fit(X, y)

        print("\nPiensa en un Pokémon...")
        exito = recorrer_arbol(clf, all_attributes, y)

        if exito:
            print("\nJugamos de nuevo?")
            if input("(s/n): ").lower() != "s":
                break

        else:
            # Agregar nuevo Pokémon
            nuevo_nombre = input("¿En qué Pokémon estabas pensando?: ").strip()
            nuevo_atributos = []

            print("Responde las siguientes preguntas sobre este Pokémon:")
            for attr in all_attributes:
                resp = input(f"¿Tu Pokémon es {attr}? (s/n): ").lower()
                if resp == "s":
                    nuevo_atributos.append(attr)

            # Preguntar si quiere agregar un atributo nuevo
            if input("¿Quieres agregar un nuevo atributo para diferenciarlo mejor? (s/n): ").lower() == "s":
                nuevo_attr = input("Escribe el nuevo atributo: ").strip()
                nuevo_atributos.append(nuevo_attr)
                all_attributes.append(nuevo_attr)
                # Actualizar todos los Pokémon en la base de datos para incluir el nuevo atributo si no lo tienen
                for p in db:
                    if nuevo_attr not in p["caracteristicas"]:
                        # No lo tiene, no lo agregamos explícitamente, la matriz lo interpreta como ausencia
                        pass

            # Guardar en la base de datos
            db.append({"pokemon": nuevo_nombre, "caracteristicas": nuevo_atributos})
            guardar_db(db)

            print(f"He aprendido sobre {nuevo_nombre}.")

# 5. Ejecutar

if __name__ == "__main__":
    jugar()
