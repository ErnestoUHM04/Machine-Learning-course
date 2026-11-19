# ML P2
# El individuo perfecto
# Ernesto Ulises Hernández Martínez
import random
import math
# El objetivo de esta practica es tener una población de 100 individuos, que se irán cruzando a lo largo de varias generaciones, tal que se busque obtener un individuo perfecto.
# Inicialmente existen 50 parejas (100 individuos), con 20 genes cada uno.
# Cada pareja tiene 2 hijos por generación, los hijos heredarán los genes de sus padres.
# Ejemplo de genes de un individuo
# [1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9][1-9]
#       Individuo perfecto
# [9][9][9][9][9][9][9][9][9][9][9][9][9][9][9][9][9][9][9][9]
# La herencia será tomar el gen i del padre y madre, promediando sus valores y asignandolos a sus hijos, cuando el valor sea .5, se asigna de manera random el número mayor a un hijo y el menor a otro

# En cada generación cada hijo nacido tiene de 10 - 25 % de probabilidad de mutar un gen, eligiendo un gen al azar y asignando un valor aleatorio entre 1 y 9.
# Es importante no cruzar a los hijos entre ellos

class Individuo:
    def __init__(self, id, genes=None, generation=0):
        self.id = id
        # tenemos que llevar el track de los padres para evitar incesto
        self.parent1 = None
        #self.parent2 = None # No es necesario
        self.genes = genes if genes is not None else [random.randint(1, 9) for _ in range(20)]
        self.generation = generation
        self.es_padre = False # Quiero saber si el individuo ya se reprodujo o no
        self.fitness = self.calcular_fitness() # Fitness será la cantidad de 9's en el individuo
        self.p_rank = 0
        # Al nacer un individuo no es padre de nadie

    def __repr__(self):
        return f"Individuo({self.id}, {self.parent1}, {self.genes}, {self.generation})"

    def calcular_fitness(self):
        #self.fitness = self.genes.count(9)
        self.fitness = sum(self.genes) / 20
        return self.fitness

    @staticmethod
    def siguiente_generacion(poblacion):
        # En esta funcion elegiremos a las parejas de la poblacion
        # La nueva población tendrá también 100 individuos
        new_poblacion = []
        new_id = 0
        poblacion = Individuo.ranking(poblacion)
        # Una vez que usamos ranking, debemos elegir a los mejores para reproducirse
        # les damos su ranking para la probabilidad de eleccion
        s = 2
        for i in range(len(poblacion)): # de 0 a 99
            poblacion[i].p_rank = ((2 - s) / 100) + (2 * i * (s - 1) / (99 * 100))
        # Una vez están ordenados y rankeados, los podremos elegir
        for i in range(len(poblacion)):
            if random.random() < poblacion[i].p_rank:
                # Esto indica que elegimos un individuo para reproducirse
                if not poblacion[i].es_padre:
                    p1 = poblacion[i]
                while True:
                    j = random.randint(0, 99)
                    if j != i and not poblacion[j].es_padre:
                        p2 = poblacion[j]
                        break
                if random.random() < poblacion[j].p_rank:
                    if not poblacion[j].es_padre:
                        p2 = poblacion[j]
                    # Cruzamos a los padres
                    Hijo1, Hijo2 = Individuo.cruzar(new_id, p1, p2)
                    new_poblacion.append(Hijo1)
                    new_poblacion.append(Hijo2)
                    p1.es_padre = True
                    p2.es_padre = True
                    new_id += 2
        return new_poblacion
    #  Habrá que devolver la lista con la nueva generación (población)

    @staticmethod
    def torneo(Participante1, Participante2, Participante3, Participante4):
        # Del random choice de 4 individuos, elegimos 2 para cruzar
        participantes = [Participante1, Participante2, Participante3, Participante4]
        participantes.sort(key=lambda x: x.fitness, reverse=True)
        # Hay que elgir a los dos más fuertes con base al Fitness
        # Retornamos los dos individuos más fuertes
        return participantes[0], participantes[1]

    @staticmethod
    def ranking(poblacion):
        # Ordenamos la población por fitness
        # 0 debe de ser el peor individuo y n el mejor
        poblacion.sort(key=lambda x: x.fitness)
        return poblacion

    @staticmethod
    def cruzar(id, Padre1, Padre2):
        # Cada pareja tendra dos hijos
        Hijo1 = Individuo(id, [], Padre1.generation + 1)
        id += 1
        Hijo2 = Individuo(id, [], Padre1.generation + 1)

        Hijo1.parent1 = Padre1.id
        #Hijo1.parent2 = Padre2.id
        Hijo2.parent1 = Padre1.id
        #Hijo2.parent2 = Padre2.id
        Hijo1.es_padre = False
        Hijo2.es_padre = False
        # Cruzamos a los padres
        for i in range(20):
            gen1 = Padre1.genes[i]
            gen2 = Padre2.genes[i]
            promedio = (gen1 + gen2) / 2
            # Si promedio es entero entonces le asignamos este valor a los hijos
            if promedio.is_integer():
                Hijo1.genes.append(int(promedio))
                Hijo2.genes.append(int(promedio))
            else: # Si no es entero, entonces asignamos el entero siguiente a un hijo y el entero anterior a otro
                # Podemos usar floor y ceil
                # Tiene que ser de manera random
                if random.random() < 0.5: # Coin flip
                    Hijo1.genes.append(math.ceil(promedio))
                    Hijo2.genes.append(math.floor(promedio))
                else:
                    Hijo1.genes.append(math.floor(promedio))
                    Hijo2.genes.append(math.ceil(promedio))
        # Esto es para la mayoría de casos, pero hay que tener en cuenta las mutaciones
        if random.random() < 0.25: # 10% de probabilidad de mutación
            i = random.randint(0, 19) # Vemos cual muta
            Hijo1.genes[i] = random.randint(1, 9)
        if random.random() < 0.25:
            i = random.randint(0, 19)
            Hijo2.genes[i] = random.randint(1, 9)
        return Hijo1, Hijo2


def main():
    # Primero tenemos que crear 100 individuos de manera aleatoria
    # Crearemos una lista de individuos, usando la clase Individuo
    poblacion = [Individuo(i) for i in range(100)]
    generacion = 0
    while True:
        print("Generacion: ", generacion)
        generacion += 1
        for individuo in poblacion:
            print(individuo)
            if individuo.genes == [9] * 20: # Equivalente a puros 9's
                print("Individuo perfecto encontrado!")
                return
        # Si no encontramos un individuo perfecto, tenemos que crear la siguiente generación
        poblacion = Individuo.siguiente_generacion(poblacion)

if __name__ == "__main__":
    main()