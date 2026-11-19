#  Aprendizaje de Máquina
#  P1 :  "20 Questions"
#  Ernesto Ulises Hernández Martínez
#################################################
#  Este programa debe de ser capaz de adivinar el pensamiento de un usuario en 20 preguntas
import numpy as np
import pandas as pd

# Cargamos el dataset con las cosas posibles de adivinar
try:
    #dataset_name = "P1/20Q_animals.csv"
    dataset_name = "P1/20Q_dataset.csv"
    dataset = pd.read_csv(dataset_name, encoding="latin1")
    original_dataset = dataset.copy()
except FileNotFoundError:
    print(f"Error: '{dataset_name}' not found. Please ensure the file exists in the correct location.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: '{dataset_name}' is empty or malformed.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: '{dataset_name}' could not be parsed. Please check the file format.")
    exit(1)


# Nodo para la lista (en cada nodo hay #1s, #0s y su coeficiente)
# Cada nodo tiene un id que representa su columna (después de la columna objeto)
class ColumnNode:
    def __init__(self, id, nombre, ones, zeros, total):
        # esto es un nodo
        self.id = id
        self.nombre = nombre
        self.ones = ones
        self.zeros = zeros
        self.total = total
        # es el caluculo de la métrica usada para la heuristica
        self.p1 = ones / self.total if self.total != 0 else 0
        self.p0 = zeros / self.total if self.total != 0 else 0
        self.E =  - (self.p1 * (np.log2(self.p1) if self.p1 > 0 else 0) + self.p0 * (np.log2(self.p0) if self.p0 > 0 else 0))

    def __repr__(self):
        return f"ColumnNode(id={self.id}, nombre={self.nombre}, ones={self.ones}, zeros={self.zeros}, total={self.total}, p1={self.p1:.2f}, p0={self.p0:.2f}, E={self.E:.2f})"

    def create_column_nodes():
        data = dataset.iloc[:, 1:]
        nodes = []
        id = 0
        for col in data.columns:
            zeros = (data[col] == 0).sum()
            ones = (data[col] == 1).sum()
            # Aqui se guarda el id, nombre, #1s y #0s
            node = ColumnNode(id, col, ones, zeros, ones + zeros)
            id += 1
            nodes.append(node)
        return nodes

    def print_column_nodes(nodes):
        for node in nodes:
            print(node)
            print("-----")

    def select_best_question(nodes):
        # Se debería usar la fórmula de la entropia para ver que pregunta divide mejor al conjunto de datos

        # Si todos los nodos tienen ratio igual a inf o 0, significa que todos los objetos son idénticos en las columnas restantes
        all_same = all(node.ones == 0 or node.zeros == 0 for node in nodes)
        if all_same:
            # No hay más preguntas útiles, se debe probar opción por opción
            return None
        # El mejor nodo, o pregunta será aquel que divida mejor a la muestra de datos
        # Se debería usar la fórmula de la entropia para ver que pregunta divide mejor al conjunto de datos
        best_node = None
        for node in nodes:
            # buscamos la Entropía E más cercana a 1
            if best_node is None or node.E > best_node.E:
                best_node = node
        return best_node

    def ask_question(column_nodes, question_number):
        global dataset, original_dataset, dataset_name
        ColumnNode.print_column_nodes(column_nodes) # Esta linea es para debug
        # La pregunta tiene que ser la que mejor divida para cada iteración
        question = ColumnNode.select_best_question(column_nodes)
        question_number += 1
        if question:
            while True:
                response = input(f"{question_number}.- ¿{question.nombre}? (s/n): ")
                if response.lower() in ["s", "n"]:
                    break
                print("Por favor responde solo 's' o 'n'.")
            if response.lower() == "s":
                dataset = dataset[dataset[question.nombre] == 1]
            else:
                dataset = dataset[dataset[question.nombre] == 0]
            column_nodes = ColumnNode.create_column_nodes()
            if len(dataset) > 1:
                ColumnNode.ask_question(column_nodes, question_number)
            elif len(dataset) == 1:
                print(f"¡Adiviné! El objeto es: {dataset.iloc[0, 0]}")
                while True:
                    final_answer = input("¿Estaba en lo correcto? (s/n): ")
                    if final_answer.lower() in ["s", "n"]:
                        break
                    print("Por favor responde solo 's' o 'n'.")
                if final_answer.lower() == "s":
                    print("¡Genial! Gracias por jugar.")
                else:
                    new_item = input("Oh no, ¿en qué estabas pensando?: ")
                    new_row = {original_dataset.columns[0]: new_item}
                    for col in original_dataset.columns[1:]:
                        val = input(f"¿{col}? (1/0): ")
                        while val not in ["0", "1"]:
                            val = input(f"Por favor responde 1 o 0. ¿{col}? (1/0): ")
                        new_row[col] = int(val)
                    original_dataset = pd.concat([original_dataset, pd.DataFrame([new_row])], ignore_index=True)
                    original_dataset.to_csv(dataset_name, index=False)
                    print("¡Nuevo objeto agregado al dataset!")
            else:
                print("No se pudo adivinar el objeto. No hay coincidencias.")
                new_item = input("¿En qué cosa estabas pensando?: ")
                new_row = {original_dataset.columns[0]: new_item}
                for col in original_dataset.columns[1:]:
                    val = input(f"¿{col}? (1/0): ")
                    while val not in ["0", "1"]:
                        val = input(f"Por favor responde 1 o 0. ¿{col}? (1/0): ")
                    new_row[col] = int(val)
                original_dataset = pd.concat([original_dataset, pd.DataFrame([new_row])], ignore_index=True)
                original_dataset.to_csv(dataset_name, index=False)
                print("¡Nuevo objeto agregado al dataset!")
        else:
            # No hay más preguntas útiles, probar opción por opción
            print("No hay más preguntas útiles. Probando opción por opción...")
            for idx, row in dataset.iterrows():
                respuesta = input(f"¿Tu objeto es '{row[dataset.columns[0]]}'? (s/n): ")
                if respuesta.lower() == "s":
                    print(f"¡Adiviné! El objeto es: {row[dataset.columns[0]]}")
                    print("¡Genial! Gracias por jugar.")
                    return
            print("No se pudo adivinar el objeto. No hay coincidencias.")
            new_item = input("¿En qué cosa estabas pensando?: ")
            new_row = {original_dataset.columns[0]: new_item}
            for col in original_dataset.columns[1:]:
                val = input(f"¿{col}? (1/0): ")
                while val not in ["0", "1"]:
                    val = input(f"Por favor responde 1 o 0. ¿{col}? (1/0): ")
                new_row[col] = int(val)
            original_dataset = pd.concat([original_dataset, pd.DataFrame([new_row])], ignore_index=True)
            original_dataset.to_csv(dataset_name, index=False)
            print("¡Nuevo objeto agregado al dataset!")


def main():
    column_nodes = ColumnNode.create_column_nodes()
    #ColumnNode.print_column_nodes(column_nodes) # Esta linea es para debug
    # Aquí se puede implementar la lógica para hacer preguntas al usuario
    # ask_question es la funcion recursiva con la que se logra la encuesta
    ColumnNode.ask_question(column_nodes, 0)

if __name__ == "__main__":
    main()
