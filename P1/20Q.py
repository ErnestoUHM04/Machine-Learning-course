#  Aprendizaje de Máquina
#  P1 :  "20 Questions"
#  Ernesto Ulises Hernández Martínez
#################################################
#  Este programa debe de ser capaz de adivinar el pensamiento de un usuario en 20 preguntas
import pandas as pd

# Cargamos el dataset con las cosas posibles de adivinar
try:
    dataset_name = "P1/20Q_animals.csv"
    dataset = pd.read_csv(dataset_name, encoding="latin1")
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
    def __init__(self, id, nombre, ones, zeros):
        # esto es un nodo
        self.id = id
        self.nombre = nombre
        self.ones = ones
        self.zeros = zeros
        # es el caluculo de la métrica usada para la heuristica
        self.ratio = ones / zeros if zeros != 0 else float('inf')

    def __repr__(self):
        return f"ColumnNode(id={self.id}, nombre={self.nombre}, ones={self.ones}, zeros={self.zeros}, ratio={self.ratio:.2f})"

    def create_column_nodes():
        data = dataset.iloc[:, 1:]
        nodes = []
        id = 0
        for col in data.columns:
            zeros = (data[col] == 0).sum()
            ones = (data[col] == 1).sum()
            # Aqui se guarda el id, nombre, #1s y #0s
            node = ColumnNode(id, col, ones, zeros)
            id += 1
            nodes.append(node)
        return nodes

    def print_column_nodes(nodes):
        for node in nodes:
            print(node)
            print("-----")

    def select_best_question(nodes):
        # El mejor nodo, o pregunta será aquel que divida mejor a la muestra de datos
        # Buscaremos que el valor ratio sea lo más cercano al 1
        best_node = None
        for node in nodes:
            if best_node is None or abs(node.ratio - 1) < abs(best_node.ratio - 1):
                best_node = node
            # Hay que pensar que hacer si solo quedan 1s o 0s
        return best_node

    def ask_question(column_nodes, question_number):
        global dataset, dataset_name
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
                # Filtramos el dataset para solo los objetos con 1 en la columna preguntada
                dataset = dataset[dataset[question.nombre] == 1]
            else:
                # Filtramos el dataset para solo los objetos con 0 en la columna preguntada
                dataset = dataset[dataset[question.nombre] == 0]
            # Recalculamos los nodos con el dataset filtrado
            column_nodes = ColumnNode.create_column_nodes()
            # Volvemos a hacer la pregunta si quedan objetos
            # Se detendrá hasta que terminemos con solo una opción, o ninguna
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
                    # Pedimos los valores para cada columna
                    new_row = {dataset.columns[0]: new_item}
                    for col in dataset.columns[1:]:
                        val = input(f"¿{col}? (1/0): ")
                        while val not in ["0", "1"]:
                            val = input(f"Por favor responde 1 o 0. ¿{col}? (1/0): ")
                        new_row[col] = int(val)
                    # Agregamos la nueva fila al dataset
                    dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)
                    # Guardamos el dataset actualizado
                    dataset.to_csv(dataset_name, index=False)
                    print("¡Nuevo objeto agregado al dataset!")
            else:
                print("No se pudo adivinar el objeto. No hay coincidencias.")
                new_item = input("¿En qué cosa estabas pensando?: ")
                new_row = {dataset.columns[0]: new_item}
                for col in dataset.columns[1:]:
                    val = input(f"¿{col}? (1/0): ")
                    while val not in ["0", "1"]:
                        val = input(f"Por favor responde 1 o 0. ¿{col}? (1/0): ")
                    new_row[col] = int(val)
                dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)
                dataset.to_csv(dataset_name, index=False)
                print("¡Nuevo objeto agregado al dataset!")


def main():
    column_nodes = ColumnNode.create_column_nodes()
    #ColumnNode.print_column_nodes(column_nodes) # Esta linea es para debug
    # Aquí se puede implementar la lógica para hacer preguntas al usuario
    # ask_question es la funcion recursiva con la que se logra la encuesta
    ColumnNode.ask_question(column_nodes, 0)

if __name__ == "__main__":
    main()
