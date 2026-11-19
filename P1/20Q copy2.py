#  Aprendizaje de Maquina
#  P1 :  "20 Questions"
#  Ernesto Ulises Hernandez Martinez
#################################################
#  Este programa debe de ser capaz de adivinar el pensamiento de un usuario en 20 preguntas

# Adaptación para leer pokemon.json
import json
import math

dataset_name = "pokemon.json"
try:
    with open(dataset_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    original_dataset = list(dataset)
except FileNotFoundError:
    print(f"Error: '{dataset_name}' not found. Por favor asegúrate de que el archivo existe.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: '{dataset_name}' está vacío o mal formado.")
    exit(1)

# Nodo para la lista (en cada nodo hay #1s, #0s y su coeficiente)
# Cada nodo tiene un id que representa su característica
class ColumnNode:
    def __init__(self, id, nombre, ones, zeros):
        self.id = id
        self.nombre = nombre
        self.ones = ones
        self.zeros = zeros
        self.total = ones + zeros
        self.p1 = ones / self.total if self.total != 0 else 0
        self.p0 = zeros / self.total if self.total != 0 else 0
        self.E =  - (self.p1 * (math.log2(self.p1) if self.p1 > 0 else 0) + self.p0 * (math.log2(self.p0) if self.p0 > 0 else 0))

    def __repr__(self):
        return f"ColumnNode(id={self.id}, nombre={self.nombre}, ones={self.ones}, zeros={self.zeros}, total={self.total}, p1={self.p1:.2f}, p0={self.p0:.2f}, E={self.E:.2f})"

    @staticmethod
    def create_column_nodes():
        # Extraer todas las características posibles
        global dataset
        all_caracs = set()
        for obj in dataset:
            all_caracs.update(obj['caracteristicas'])
        nodes = []
        id = 0
        for carac in all_caracs:
            ones = sum(carac in obj['caracteristicas'] for obj in dataset)
            zeros = len(dataset) - ones
            node = ColumnNode(id, carac, ones, zeros)
            id += 1
            nodes.append(node)
        return nodes

    @staticmethod
    def print_column_nodes(nodes):
        for node in nodes:
            print(node)
            print("-----")


    @staticmethod
    def select_best_question(nodes):
        # Si todos los nodos tienen ratio igual a inf o 0, significa que todos los objetos son idénticos en las columnas restantes
        all_same = all(node.ones == 0 or node.zeros == 0 for node in nodes)
        if all_same:
            return None
        best_node = None
        for node in nodes:
            if best_node is None or node.E > best_node.E:
                best_node = node
        return best_node


    @staticmethod
    def ask_question(column_nodes, question_number):
        global dataset, original_dataset, dataset_name
        #ColumnNode.print_column_nodes(column_nodes) # debug opcional
        question = ColumnNode.select_best_question(column_nodes)
        question_number += 1
        if question:
            while True:
                response = input(f"{question_number}.- ¿{question.nombre}? (s/n): ")
                if response.lower() in ["s", "n"]:
                    break
                print("Por favor responde solo 's' o 'n'.")
            if response.lower() == "s":
                dataset = [obj for obj in dataset if question.nombre in obj['caracteristicas']]
            else:
                dataset = [obj for obj in dataset if question.nombre not in obj['caracteristicas']]
            column_nodes = ColumnNode.create_column_nodes()
            if len(dataset) > 1:
                ColumnNode.ask_question(column_nodes, question_number)
            elif len(dataset) == 1:
                print(f"¡Adiviné! El Pokémon es: {dataset[0]['pokemon']}")
                while True:
                    final_answer = input("¿Estaba en lo correcto? (s/n): ")
                    if final_answer.lower() in ["s", "n"]:
                        break
                    print("Por favor responde solo 's' o 'n'.")
                if final_answer.lower() == "s":
                    print("¡Genial! Gracias por jugar.")
                else:
                    new_item = input("Oh no, ¿en qué Pokémon estabas pensando?: ")
                    # Agregar nuevo Pokémon
                    all_caracs = set()
                    for obj in original_dataset:
                        all_caracs.update(obj['caracteristicas'])
                    new_caracs = []
                    for carac in all_caracs:
                        val = input(f"¿{carac}? (s/n): ")
                        while val not in ["s", "n"]:
                            val = input(f"Por favor responde 's' o 'n'. ¿{carac}? (s/n): ")
                        if val == "s":
                            new_caracs.append(carac)
                    original_dataset.append({"pokemon": new_item, "caracteristicas": new_caracs})
                    with open(dataset_name, 'w', encoding='utf-8') as f:
                        json.dump(original_dataset, f, ensure_ascii=False, indent=4)
                    print("¡Nuevo Pokémon agregado al dataset!")
            else:
                print("No se pudo adivinar el Pokémon. No hay coincidencias.")
                new_item = input("¿En qué Pokémon estabas pensando?: ")
                all_caracs = set()
                for obj in original_dataset:
                    all_caracs.update(obj['caracteristicas'])
                new_caracs = []
                for carac in all_caracs:
                    val = input(f"¿{carac}? (s/n): ")
                    while val not in ["s", "n"]:
                        val = input(f"Por favor responde 's' o 'n'. ¿{carac}? (s/n): ")
                    if val == "s":
                        new_caracs.append(carac)
                original_dataset.append({"pokemon": new_item, "caracteristicas": new_caracs})
                with open(dataset_name, 'w', encoding='utf-8') as f:
                    json.dump(original_dataset, f, ensure_ascii=False, indent=4)
                print("¡Nuevo Pokémon agregado al dataset!")
        else:
            print("No hay más preguntas útiles. Probando opción por opción...")
            for obj in dataset:
                respuesta = input(f"¿Tu Pokémon es '{obj['pokemon']}'? (s/n): ")
                if respuesta.lower() == "s":
                    print(f"¡Adiviné! El Pokémon es: {obj['pokemon']}")
                    print("¡Genial! Gracias por jugar.")
                    return
            print("No se pudo adivinar el Pokémon. No hay coincidencias.")
            new_item = input("¿En qué Pokémon estabas pensando?: ")
            all_caracs = set()
            for obj in original_dataset:
                all_caracs.update(obj['caracteristicas'])
            new_caracs = []
            for carac in all_caracs:
                val = input(f"¿{carac}? (s/n): ")
                while val not in ["s", "n"]:
                    val = input(f"Por favor responde 's' o 'n'. ¿{carac}? (s/n): ")
                if val == "s":
                    new_caracs.append(carac)
            original_dataset.append({"pokemon": new_item, "caracteristicas": new_caracs})
            with open(dataset_name, 'w', encoding='utf-8') as f:
                json.dump(original_dataset, f, ensure_ascii=False, indent=4)
            print("¡Nuevo Pokémon agregado al dataset!")



def main():
    column_nodes = ColumnNode.create_column_nodes()
    ColumnNode.ask_question(column_nodes, 0)

if __name__ == "__main__":
    main()
