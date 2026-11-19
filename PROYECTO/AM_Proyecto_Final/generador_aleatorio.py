import numpy as np
import random
from collections import deque
import os


class GeneradorEscenariosAleatorios:
    """
    Generador de escenarios aleatorios para entrenamiento por época
    Cada época genera 12 nuevos escenarios RESOLUBLES y únicos
    """

    # Símbolos del Mapa
    WALL = "#"
    EMPTY = "'"
    START = "S"
    EXIT_R = "R"
    TREASURE = "T"
    ENEMY_X = "X"
    ENEMY_E = "E"
    SHOP = "P"

    # Pesos para generación de mapas (NO afectan entrenamiento, solo BFS)
    # Los rewards reales están en agente1.py en el método step()
    PESOS = {
        WALL: -100,  # Penalización alta (imposible atravesar)
        EMPTY: 0,  # Neutral (actualizado: antes -1)
        START: 0,  # Posición inicial
        EXIT_R: 200,  # Meta principal (actualizado: antes 100)
        TREASURE: 50,  # Recompensa por tesoro
        ENEMY_X: -30,  # Penalización por enemigo débil
        ENEMY_E: -50,  # Penalización por enemigo fuerte
        SHOP: 10,  # Recompensa por tienda
    }

    MAP_SIZE = 18

    def __init__(self, seed=None):
        """
        Inicializa el generador
        Args:
            seed: Semilla para reproducibilidad (opcional)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def crear_mapa_vacio(self):
        """Crea un mapa base con bordes de paredes"""
        mapa = [
            [self.EMPTY for _ in range(self.MAP_SIZE)] for _ in range(self.MAP_SIZE)
        ]

        # Bordes
        for i in range(self.MAP_SIZE):
            mapa[0][i] = self.WALL
            mapa[self.MAP_SIZE - 1][i] = self.WALL
            mapa[i][0] = self.WALL
            mapa[i][self.MAP_SIZE - 1] = self.WALL

        return mapa

    def es_posicion_valida(self, mapa, fila, col):
        """Verifica si una posición es válida y está vacía"""
        if 0 <= fila < self.MAP_SIZE and 0 <= col < self.MAP_SIZE:
            return mapa[fila][col] == self.EMPTY
        return False

    def bfs_camino_existe(self, mapa, start_pos, end_pos):
        """
        Verifica si existe un camino entre start_pos y end_pos usando BFS
        Retorna True si hay camino, False si no
        """
        if not start_pos or not end_pos:
            return False

        visitados = set()
        cola = deque([start_pos])
        visitados.add(start_pos)

        direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while cola:
            fila, col = cola.popleft()

            if (fila, col) == end_pos:
                return True

            for df, dc in direcciones:
                nueva_fila, nueva_col = fila + df, col + dc

                if (nueva_fila, nueva_col) not in visitados:
                    if (
                        0 <= nueva_fila < self.MAP_SIZE
                        and 0 <= nueva_col < self.MAP_SIZE
                    ):
                        celda = mapa[nueva_fila][nueva_col]
                        # Puede pasar por cualquier cosa excepto paredes
                        if celda != self.WALL:
                            visitados.add((nueva_fila, nueva_col))
                            cola.append((nueva_fila, nueva_col))

        return False

    def generar_laberinto_recursivo(self, mapa, x1, y1, x2, y2, densidad=0.3):
        """
        Genera un laberinto usando división recursiva
        densidad: probabilidad de añadir paredes (0.0 - 1.0)
        """
        if x2 - x1 < 3 or y2 - y1 < 3:
            return

        if random.random() > densidad:
            return

        # Elegir orientación
        horizontal = random.choice([True, False])

        if horizontal:
            # Dividir horizontalmente
            y = random.randint(y1 + 1, y2 - 1)
            for x in range(x1, x2 + 1):
                if mapa[y][x] == self.EMPTY:
                    mapa[y][x] = self.WALL

            # Hacer un hueco
            hueco = random.randint(x1, x2)
            mapa[y][hueco] = self.EMPTY

            # Recursión
            self.generar_laberinto_recursivo(mapa, x1, y1, x2, y - 1, densidad)
            self.generar_laberinto_recursivo(mapa, x1, y + 1, x2, y2, densidad)
        else:
            # Dividir verticalmente
            x = random.randint(x1 + 1, x2 - 1)
            for y in range(y1, y2 + 1):
                if mapa[y][x] == self.EMPTY:
                    mapa[y][x] = self.WALL

            # Hacer un hueco
            hueco = random.randint(y1, y2)
            mapa[hueco][x] = self.EMPTY

            # Recursión
            self.generar_laberinto_recursivo(mapa, x1, y1, x - 1, y2, densidad)
            self.generar_laberinto_recursivo(mapa, x + 1, y1, x2, y2, densidad)

    def generar_habitaciones(self, mapa, num_habitaciones=4):
        """Genera habitaciones rectangulares aleatorias"""
        for _ in range(num_habitaciones):
            ancho = random.randint(3, 6)
            alto = random.randint(3, 6)

            x = random.randint(2, self.MAP_SIZE - ancho - 2)
            y = random.randint(2, self.MAP_SIZE - alto - 2)

            # Crear habitación
            for i in range(y, min(y + alto, self.MAP_SIZE - 1)):
                for j in range(x, min(x + ancho, self.MAP_SIZE - 1)):
                    if i == y or i == y + alto - 1 or j == x or j == x + ancho - 1:
                        if random.random() > 0.3:  # 70% probabilidad de pared
                            mapa[i][j] = self.WALL
                    else:
                        mapa[i][j] = self.EMPTY

    def colocar_elemento(self, mapa, simbolo, cantidad, evitar_posiciones=None):
        """
        Coloca elementos en posiciones vacías del mapa
        """
        if evitar_posiciones is None:
            evitar_posiciones = set()

        colocados = 0
        intentos = 0
        max_intentos = cantidad * 50

        while colocados < cantidad and intentos < max_intentos:
            fila = random.randint(1, self.MAP_SIZE - 2)
            col = random.randint(1, self.MAP_SIZE - 2)

            if (fila, col) not in evitar_posiciones and mapa[fila][col] == self.EMPTY:
                mapa[fila][col] = simbolo
                evitar_posiciones.add((fila, col))
                colocados += 1

            intentos += 1

        return colocados

    def generar_escenario_aleatorio(self, dificultad=1, tipo_mapa="mixto"):
        """
        Genera un escenario aleatorio RESOLUBLE

        Args:
            dificultad: 1-12 (afecta cantidad de enemigos y complejidad)
            tipo_mapa: "abierto", "laberinto", "habitaciones", "mixto"

        Returns:
            lista de strings representando el mapa
        """
        max_intentos_resolucion = 20

        for intento in range(max_intentos_resolucion):
            # Crear mapa base
            mapa = self.crear_mapa_vacio()

            # Generar estructura según tipo
            if tipo_mapa == "laberinto" or (
                tipo_mapa == "mixto" and random.random() < 0.3
            ):
                densidad = 0.2 + (dificultad / 12) * 0.3  # 0.2 a 0.5
                self.generar_laberinto_recursivo(
                    mapa, 1, 1, self.MAP_SIZE - 2, self.MAP_SIZE - 2, densidad
                )

            elif tipo_mapa == "habitaciones" or (
                tipo_mapa == "mixto" and random.random() < 0.3
            ):
                num_habitaciones = min(3 + dificultad // 3, 8)
                self.generar_habitaciones(mapa, num_habitaciones)

            # Colocar START y EXIT
            # START siempre arriba-izquierda
            start_zona = [
                (i, j)
                for i in range(1, 4)
                for j in range(1, 4)
                if mapa[i][j] == self.EMPTY
            ]

            if not start_zona:
                continue

            start_pos = random.choice(start_zona)
            mapa[start_pos[0]][start_pos[1]] = self.START

            # EXIT siempre abajo-derecha
            exit_zona = [
                (i, j)
                for i in range(self.MAP_SIZE - 4, self.MAP_SIZE - 1)
                for j in range(self.MAP_SIZE - 4, self.MAP_SIZE - 1)
                if mapa[i][j] == self.EMPTY
            ]

            if not exit_zona:
                continue

            exit_pos = random.choice(exit_zona)
            mapa[exit_pos[0]][exit_pos[1]] = self.EXIT_R

            # VERIFICAR QUE EXISTE CAMINO
            if not self.bfs_camino_existe(mapa, start_pos, exit_pos):
                continue  # Regenerar si no hay camino

            # Posiciones ocupadas
            ocupadas = {start_pos, exit_pos}

            # Calcular cantidades según dificultad
            num_enemigos_x = 2 + dificultad
            num_enemigos_e = dificultad // 3
            num_tesoros = 2 + dificultad // 2
            num_tiendas = 1 + dificultad // 4

            # Colocar elementos
            self.colocar_elemento(mapa, self.ENEMY_X, num_enemigos_x, ocupadas)
            self.colocar_elemento(mapa, self.ENEMY_E, num_enemigos_e, ocupadas)
            self.colocar_elemento(mapa, self.TREASURE, num_tesoros, ocupadas)
            self.colocar_elemento(mapa, self.SHOP, num_tiendas, ocupadas)

            # Convertir a lista de strings
            mapa_strings = ["".join(fila) for fila in mapa]

            return mapa_strings

        # Si después de muchos intentos no se logró, retornar mapa simple
        print(
            f"Advertencia: No se pudo generar mapa resoluble para dificultad {dificultad}, usando mapa simple"
        )
        return self.generar_mapa_simple()

    def generar_mapa_simple(self):
        """Genera un mapa simple garantizado resoluble (respaldo)"""
        mapa = self.crear_mapa_vacio()

        # Camino directo diagonal
        mapa[1][1] = self.START
        mapa[self.MAP_SIZE - 2][self.MAP_SIZE - 2] = self.EXIT_R

        # Camino libre
        for i in range(1, self.MAP_SIZE - 1):
            for j in range(1, self.MAP_SIZE - 1):
                if abs(i - j) <= 2:
                    if mapa[i][j] == self.EMPTY:
                        if random.random() < 0.2:
                            mapa[i][j] = self.ENEMY_X

        return ["".join(fila) for fila in mapa]

    def generar_12_escenarios_epoca(self, epoca=0):
        """
        Genera 12 escenarios aleatorios para una época de entrenamiento
        Distribución uniforme: 3 abiertos, 3 laberintos, 3 habitaciones, 3 mixtos

        Args:
            epoca: Número de época (afecta semilla y dificultad progresiva)

        Returns:
            lista con 12 mapas (cada uno es una lista de strings)
        """
        escenarios = []

        # DISTRIBUCIÓN UNIFORME: 3 de cada tipo
        tipos_mapa = (
            ["abierto"] * 3 + ["laberinto"] * 3 + ["habitaciones"] * 3 + ["mixto"] * 3
        )
        random.shuffle(tipos_mapa)  # Mezclar orden para variedad

        print(f"\n{'='*60}")
        print(f"  GENERANDO 12 ESCENARIOS PARA ÉPOCA {epoca}")
        print(f"  Distribución: 3 abiertos, 3 laberintos, 3 habitaciones, 3 mixtos")
        print(f"{'='*60}")

        for i in range(12):
            dificultad = i + 1  # Dificultad 1-12
            tipo = tipos_mapa[i]  # Usar tipo de la lista mezclada

            print(
                f"  Escenario {i+1}/12 - Dificultad: {dificultad} - Tipo: {tipo}...",
                end=" ",
            )

            escenario = self.generar_escenario_aleatorio(dificultad, tipo)
            escenarios.append(escenario)

            print("✓")

        print(f"{'='*60}\n")
        return escenarios

    def guardar_escenarios_epoca(
        self, epoca, escenarios, carpeta="mapas_entrenamiento"
    ):
        """
        Guarda los 12 escenarios de una época en archivos

        Args:
            epoca: Número de época
            escenarios: Lista de 12 mapas
            carpeta: Carpeta donde guardar
        """
        carpeta_epoca = os.path.join(carpeta, f"epoca_{epoca:04d}")

        if not os.path.exists(carpeta_epoca):
            os.makedirs(carpeta_epoca)

        for idx, escenario in enumerate(escenarios, 1):
            nombre_archivo = os.path.join(carpeta_epoca, f"escenario_{idx:02d}.txt")
            with open(nombre_archivo, "w", encoding="utf-8") as f:
                for linea in escenario:
                    f.write(linea + "\n")

        print(f"Época {epoca}: 12 escenarios guardados en '{carpeta_epoca}'")

    def obtener_peso(self, simbolo):
        """Devuelve el peso/recompensa de un símbolo"""
        return self.PESOS.get(simbolo, 0)

    def crear_matriz_pesos(self, escenario):
        """Convierte un escenario en matriz de pesos para Q-learning"""
        matriz = []
        for linea in escenario:
            fila = [self.obtener_peso(simbolo) for simbolo in linea]
            matriz.append(fila)
        return np.array(matriz)

    def visualizar_escenario(self, escenario, titulo="ESCENARIO"):
        """Muestra un escenario en consola"""
        print(f"\n{'='*40}")
        print(f"{titulo}")
        print(f"{'='*40}")
        for linea in escenario:
            print(linea)
        print(f"{'='*40}\n")

    def verificar_resolubles(self, escenarios):
        """
        Verifica que todos los escenarios sean resolubles
        Retorna (resolubles, no_resolubles)
        """
        resolubles = []
        no_resolubles = []

        for idx, escenario in enumerate(escenarios):
            # Convertir a matriz
            mapa = [list(linea) for linea in escenario]

            # Encontrar S y R
            start_pos = None
            exit_pos = None

            for i in range(self.MAP_SIZE):
                for j in range(self.MAP_SIZE):
                    if mapa[i][j] == self.START:
                        start_pos = (i, j)
                    elif mapa[i][j] == self.EXIT_R:
                        exit_pos = (i, j)

            # Verificar camino
            if self.bfs_camino_existe(mapa, start_pos, exit_pos):
                resolubles.append(idx)
            else:
                no_resolubles.append(idx)

        return resolubles, no_resolubles


# ==========================================
# EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print(" GENERADOR DE ESCENARIOS ALEATORIOS POR ÉPOCA")
    print("=" * 60)

    # Crear generador con semilla para reproducibilidad (opcional)
    generador = GeneradorEscenariosAleatorios(seed=42)

    # Ejemplo: Generar escenarios para 3 épocas
    num_epocas_ejemplo = 3

    print(f"\nGenerando escenarios para {num_epocas_ejemplo} épocas de ejemplo...\n")

    for epoca in range(num_epocas_ejemplo):
        # Generar 12 escenarios aleatorios
        escenarios = generador.generar_12_escenarios_epoca(epoca)

        # Verificar que sean resolubles
        resolubles, no_resolubles = generador.verificar_resolubles(escenarios)
        print(f"   ✓ Época {epoca}: {len(resolubles)}/12 escenarios resolubles")

        if no_resolubles:
            print(f"Escenarios NO resolubles: {no_resolubles}")

        # Guardar escenarios
        generador.guardar_escenarios_epoca(epoca, escenarios)

        # Visualizar primer escenario de cada época
        if epoca == 0:
            generador.visualizar_escenario(
                escenarios[0], f"EJEMPLO: Época {epoca} - Escenario 1"
            )

    print("\n" + "=" * 60)
    print(" USO EN ENTRENAMIENTO")
    print("=" * 60)
    print(
        """
# En tu loop de entrenamiento:

generador = GeneradorEscenariosAleatorios()

for epoca in range(num_epocas):
    # Generar 12 nuevos escenarios aleatorios
    escenarios = generador.generar_12_escenarios_epoca(epoca)
    
    # Entrenar el agente con estos 12 escenarios
    for idx, escenario in enumerate(escenarios):
        # Convertir a matriz de pesos si es necesario
        matriz_pesos = generador.crear_matriz_pesos(escenario)
        
        # Tu lógica de entrenamiento aquí
        # ...
    
    print(f"Época {epoca} completada")
"""
    )

    print("\n¡Proceso completado!")
    print(f"\nLos escenarios se guardaron en: mapas_entrenamiento/epoca_XXXX/")
