import numpy as np
import random
import time
import os
import glob
import pickle
import json
import heapq
from datetime import datetime

from logica_combate.logica_combate import SistemaCombate
from logica_objetos.logica_objetos import ObjetoLogico
from generador_aleatorio import GeneradorEscenariosAleatorios


# Función auxiliar para crear caña inicial
def crear_cana_basica_inicial():
    """Crea una caña básica con stats mínimas para empezar"""
    config = {
        "rareza": (1, 1),
        "nivel_requerido": (1, 1),
        "precio_venta": 40,
        "porcentaje_completado": (0, 1),
        "valor_estrategico": (1, 2),
    }
    cana = ObjetoLogico(config, usar_rareza_ponderada=False, predecir_automatico=False)
    cana.datos["tipo_objeto"] = "cana_basica"
    return cana


# ==========================================
# PATHFINDING CON A*
# ==========================================


class PathFinder:
    """Algoritmo A* optimizado para encontrar rutas óptimas en el mapa"""

    def __init__(self):
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # Norte, Sur, Este, Oeste
        self.direction_to_action = {(-1, 0): 0, (1, 0): 1, (0, 1): 2, (0, -1): 3}
        self.max_search_distance = 30  # Límite de búsqueda para optimización
        self.cache = {}  # Cache de rutas calculadas
        self.cache_max_size = 1000  # Máximo de entradas en cache

    def heuristic(self, pos, goal):
        """Distancia Manhattan como heurística"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def is_walkable(self, cell):
        """Determina si una celda es transitable"""
        return cell not in ["#"]  # Todo excepto paredes es transitable

    def find_path(self, mapa, start, goal, agent_gold=0, gold_req=0):
        """
        Encuentra el camino más corto usando A* OPTIMIZADO

        Args:
            mapa: Grid del mapa actual
            start: Posición inicial (y, x)
            goal: Posición objetivo (y, x)
            agent_gold: Oro actual del agente
            gold_req: Oro requerido para salir

        Returns:
            Lista de acciones (0-3) o None si no hay camino
        """
        if start == goal:
            return []

        # OPTIMIZACIÓN: Cache de rutas
        cache_key = (start, goal)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # OPTIMIZACIÓN: Early exit si está muy lejos
        initial_distance = self.heuristic(start, goal)
        if initial_distance > self.max_search_distance:
            return None

        # Cola de prioridad: (f_score, g_score, posición, camino)
        heap = [(initial_distance, 0, start, [])]
        visited = set()
        limit = len(mapa)
        max_iterations = 200  # Límite de iteraciones para evitar búsquedas largas
        iterations = 0

        while heap and iterations < max_iterations:
            iterations += 1
            f_score, g_score, current, path = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            # Si llegamos al objetivo
            if current == goal:
                # Guardar en cache
                if len(self.cache) < self.cache_max_size:
                    self.cache[cache_key] = path
                return path

            # OPTIMIZACIÓN: Skip si ya exploramos mucho
            if g_score > self.max_search_distance:
                continue

            y, x = current

            # Explorar vecinos
            for dy, dx in self.directions:
                ny, nx = y + dy, x + dx

                # Verificar límites
                if not (0 <= ny < limit and 0 <= nx < limit):
                    continue

                cell = mapa[ny, nx]

                # Verificar si es transitable
                if not self.is_walkable(cell):
                    continue

                # Penalizar enemigos
                step_cost = 1
                if cell in ["X", "E"]:
                    step_cost = 3

                new_g = g_score + step_cost
                new_pos = (ny, nx)

                if new_pos in visited:
                    continue

                # Calcular f = g + h
                h = self.heuristic(new_pos, goal)
                new_f = new_g + h

                # Agregar acción al camino
                action = self.direction_to_action[(dy, dx)]
                new_path = path + [action]

                heapq.heappush(heap, (new_f, new_g, new_pos, new_path))

        return None  # No se encontró camino

    def find_nearest_target(
        self, mapa, start, target_symbols, agent_gold=0, gold_req=0
    ):
        """
        Encuentra el objetivo más cercano de un conjunto de símbolos (OPTIMIZADO)

        Args:
            mapa: Grid del mapa
            start: Posición inicial
            target_symbols: Lista de símbolos a buscar (ej: ['T', 'P', 'R'])
            agent_gold: Oro actual
            gold_req: Oro requerido

        Returns:
            (camino, símbolo_encontrado) o (None, None)
        """
        best_path = None
        best_length = float("inf")
        best_symbol = None

        # OPTIMIZACIÓN: Recolectar y ordenar objetivos por distancia Manhattan
        targets = []
        for symbol in target_symbols:
            positions = np.where(mapa == symbol)
            for i in range(len(positions[0])):
                goal = (positions[0][i], positions[1][i])
                distance = self.heuristic(start, goal)
                targets.append((distance, goal, symbol))

        # Ordenar por distancia y solo evaluar los 3 más cercanos
        targets.sort(key=lambda x: x[0])
        max_targets = min(3, len(targets))  # Solo los 3 más cercanos

        for _, goal, symbol in targets[:max_targets]:
            path = self.find_path(mapa, start, goal, agent_gold, gold_req)

            if path and len(path) < best_length:
                best_path = path
                best_length = len(path)
                best_symbol = symbol

                # Early exit si encontramos uno muy cercano
                if best_length <= 5:
                    break

        return best_path, best_symbol

    def get_next_action(self, mapa, start, agent_gold, gold_req):
        """
        Decide la siguiente acción basada en A*

        Estrategia:
        1. Si tiene oro suficiente -> ir a salida
        2. Si no tiene oro suficiente -> ir a tesoro más cercano
        3. Si no hay tesoros -> explorar hacia salida

        Returns:
            Acción (0-3) o None si no puede decidir
        """
        # Estrategia 1: Si tiene oro suficiente, ir a la salida
        if agent_gold >= gold_req:
            path, _ = self.find_nearest_target(
                mapa, start, ["R", "G"], agent_gold, gold_req
            )
            if path and len(path) > 0:
                return path[0]

        # Estrategia 2: Buscar tesoros
        path, _ = self.find_nearest_target(mapa, start, ["T"], agent_gold, gold_req)
        if path and len(path) > 0:
            return path[0]

        # Estrategia 3: Si no hay tesoros, ir hacia la salida (aunque no tenga oro)
        path, _ = self.find_nearest_target(
            mapa, start, ["R", "G"], agent_gold, gold_req
        )
        if path and len(path) > 0:
            return path[0]

        # Estrategia 4: Explorar hacia tienda si está disponible
        path, _ = self.find_nearest_target(mapa, start, ["P"], agent_gold, gold_req)
        if path and len(path) > 0:
            return path[0]

        return None


# Configuraciones de objetos (del dataset de entrenamiento)
OBJETOS_CONFIG = {
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
        "porcentaje_completado": (3, 3),
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

# 1. CONFIGURACIÓN Y REGLAS


class GameConfig:
    MAP_SIZE = 18

    # Símbolos del Mapa
    WALL = "#"
    EMPTY = " "
    START = "S"
    EXIT_R = "R"
    TREASURE = "T"
    ENEMY_X = "X"
    ENEMY_E = "E"
    SHOP = "P"

    @staticmethod
    def get_gold_requirement(level_idx):
        # Oro ACUMULATIVO total - Proporcional a objetos (cañas épicas/legendarias = 1000-2000 oro)
        reqs = [
            int(40 * 2 / 3),  # Mapa 0: ~26
            40,  # Mapa 1: 40
            int(160 * 1 / 3),  # Mapa 2: ~53
            int(160 * 2 / 3),  # Mapa 3: ~106
            160,  # Mapa 4
            int(640 * 1 / 3),  # Mapa 5: ~213
            int(640 * 2 / 3),  # Mapa 6: ~426
            640,  # Mapa 7
            int(2560 * 1 / 3),  # Mapa 8
            int(2560 * 2 / 3),  # Mapa 9
            2560,  # Mapa 10
            5120,  # Mapa 11+
        ]
        if level_idx < len(reqs):
            return reqs[level_idx]
        return 999999


# 2. ENTORNO DE JUEGO


class MapEnvironment:

    def __init__(self, maps_folder="mapa", use_random=False, verbose=True):
        self.maps_folder = maps_folder
        self.use_random = use_random
        self.verbose = verbose  # Control de prints
        self.map_templates = self._load_maps()
        self.maps = []

        # Rutas corregidas para los modelos
        combate_model_path = os.path.join("logica_combate", "modelo_rf.pkl")
        combate_dataset_path = os.path.join(
            "logica_combate", "dataset_peces_enemigos.csv"
        )
        self.combate = SistemaCombate(
            model_path=combate_model_path, dataset_path=combate_dataset_path
        )

        self.current_map_idx = 0
        self.agent_pos = (1, 1)
        self.max_steps = 1000  # Más tiempo para explorar
        self.steps = 0

    def _load_maps(self):
        if not os.path.exists(self.maps_folder):
            os.makedirs(self.maps_folder)

        files = sorted(glob.glob(os.path.join(self.maps_folder, "*.txt")))
        loaded_maps = []

        if self.verbose:
            print(f"Cargando mapas desde {self.maps_folder}...")

        # USAR EL TAMAÑO DE LA CONFIGURACIÓN
        SIZE = GameConfig.MAP_SIZE

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [
                    line.replace("\n", "").replace("\r", "")
                    for line in f.readlines()
                    if line.strip()
                ]

            # 1. Ajustar ALTO (Filas)
            if len(lines) < SIZE:
                lines += ["#" * SIZE] * (SIZE - len(lines))
            elif len(lines) > SIZE:
                lines = lines[:SIZE]  # Recortar si sobra

            grid_list = []
            for line in lines:
                # 2. Ajustar ANCHO (Columnas) con GameConfig.MAP_SIZE
                # ljust rellena con '#', slicing [:SIZE] corta lo que sobra
                l = list(line.ljust(SIZE, "#"))[:SIZE]
                grid_list.append(l)

            loaded_maps.append(np.array(grid_list))
            if self.verbose:
                print(f" -> {os.path.basename(file_path)} cargado ({SIZE}x{SIZE}).")

        if not loaded_maps:
            raise ValueError("No se encontraron mapas válidos.")

        return loaded_maps

    def reset(self, keep_progress=False):
        """
        keep_progress=True: Avanza de nivel
        keep_progress=False: Reinicia desde mapa 0
        """
        if not keep_progress:
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]

        if self.current_map_idx >= len(self.maps):
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]

        # BUSCAR POSICIÓN 'S' CORRECTAMENTE
        current_map = self.maps[self.current_map_idx]
        start_positions = np.where(current_map == "S")

        if len(start_positions[0]) > 0:
            self.agent_pos = (start_positions[0][0], start_positions[1][0])

        self.steps = 0
        return self.get_vision()

    def get_vision(self):
        """Extrae ventana 5x5"""
        y, x = self.agent_pos
        current_map = self.maps[self.current_map_idx]

        padded = np.pad(current_map, 2, constant_values="#")
        py, px = y + 2, x + 2

        return padded[py - 2 : py + 3, px - 2 : px + 3]

    def step(self, action, agent_gold, agent_nivel, agent_cana_rareza):
        """
        agent_nivel: nivel del agente (1-100) para verificar nivel_requerido
        agent_cana_rareza: rareza de la caña equipada (1-5) para combate
        """
        self.steps += 1
        y, x = self.agent_pos

        moves = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        dy, dx = moves[action]
        ny, nx = y + dy, x + dx

        current_map = self.maps[self.current_map_idx]
        limit = GameConfig.MAP_SIZE

        if not (0 <= ny < limit and 0 <= nx < limit):
            ny, nx = y, x

        cell = current_map[ny, nx]

        reward = (
            0.5  # BALANCEADO: Recompensa por movimiento exitoso (incentiva exploración)
        )
        done = False
        info = {}

        # Recompensa adicional si puede completar el mapa y está cerca de la salida
        req = GameConfig.get_gold_requirement(self.current_map_idx)
        if agent_gold >= req:
            # Buscar la salida más cercana
            exit_positions = np.where((current_map == "R") | (current_map == "G"))
            if len(exit_positions[0]) > 0:
                # Calcular distancia Manhattan a la salida más cercana
                min_dist = float("inf")
                for ey, ex in zip(exit_positions[0], exit_positions[1]):
                    dist = abs(ny - ey) + abs(nx - ex)
                    min_dist = min(min_dist, dist)

                # Recompensa inversamente proporcional a la distancia
                if min_dist < 10:
                    reward += (10 - min_dist) * 5  # Bonus aumentado por acercarse

        if cell == GameConfig.WALL:
            reward = -3  # BALANCEADO: -3 por pared (6x el movimiento válido)
            ny, nx = y, x

        elif cell == GameConfig.TREASURE:
            # Generar objeto aleatorio usando el modelo
            config_elegida = random.choice(list(OBJETOS_CONFIG.values()))
            objeto = ObjetoLogico(
                config_elegida, usar_rareza_ponderada=True, predecir_automatico=True
            )

            info["objeto_encontrado"] = objeto

            # Verificar si el agente tiene nivel suficiente para recogerlo
            if agent_nivel < objeto.nivel_requerido:
                reward = (
                    -10
                )  # AUMENTADO: Penalización más fuerte por nivel insuficiente
                info["objeto_rechazado"] = "nivel_insuficiente"
                # NO limpiar casilla - el tesoro permanece visible
                ny, nx = y, x  # No moverse sobre el tesoro
            else:
                # Recompensa basada en el valor estratégico del objeto
                reward = objeto.valor_estrategico * 5
                info["objeto_recogido"] = True
                # Limpiar casilla solo si se recogió exitosamente
                current_map[ny, nx] = GameConfig.EMPTY

                tipo = objeto.tipo_objeto

        elif cell in [GameConfig.ENEMY_X, GameConfig.ENEMY_E]:
            # Solo procesar combate si nos estamos moviendo HACIA el enemigo (no si ya estamos sobre él)
            old_cell = current_map[y, x]
            if old_cell not in [GameConfig.ENEMY_X, GameConfig.ENEMY_E]:
                # 1. Generar el pez acorde a la zona (Usa la lógica de amenazas 1-2, 3-4, 5-6)
                pez = self.combate.generar_enemigo_por_zona(self.current_map_idx)

                # 2. Procesar el combate usando la rareza de la caña equipada
                res = self.combate.procesar_encuentro(pez, agent_cana_rareza)

                if res["resultado"] == "victoria":
                    # Recompensa estándar del dataset
                    oro_ganado = int(res["recompensa"])
                    reward = 100 + oro_ganado  # Gran recompensa base + oro
                    info["gold_gain"] = oro_ganado
                    info["exp_gain"] = res["experiencia"]
                    # El enemigo desaparece si es derrotado
                    current_map[ny, nx] = GameConfig.EMPTY
                else:
                    # Derrota: Penalización pero PERMITIR movimiento (no quedar atascado)
                    info["gold_gain"] = 0  # ORO INTACTO
                    reward = (
                        -8
                    )  # BALANCEADO: Similar a jugar.py (evitar pero no devastador)
                    # CAMBIO: Limpiar enemigo para evitar loops, el agente puede pasar
                    current_map[ny, nx] = GameConfig.EMPTY
            else:
                # Ya estamos sobre un enemigo (segunda pasada), solo moverse
                current_map[ny, nx] = GameConfig.EMPTY

        elif cell == GameConfig.SHOP:
            # SOLO vender inventario - NO comprar nada
            info["venta_realizada"] = True
            reward = 5  # Recompensa por llegar a la tienda y vender
            # El agente puede moverse libremente sobre la tienda (no se bloquea)

        elif cell == GameConfig.EXIT_R:
            # Verificar si cumple requisitos para pasar de nivel
            req = GameConfig.get_gold_requirement(self.current_map_idx)
            if agent_gold >= req:
                # ¡Puede avanzar!
                info["level_complete"] = True
                reward = 200  # Gran recompensa por completar nivel
                done = True
            else:
                # No tiene suficiente oro
                reward = -10
                info["oro_insuficiente"] = True
                ny, nx = y, x  # No puede pasar
        elif cell == GameConfig.START:
            reward = -1  # Penalización leve por volver al inicio

        self.agent_pos = (ny, nx)

        if self.steps >= self.max_steps:
            done = True
            info["timeout"] = True

        return self.get_vision(), reward, done, info


# 3. CODIFICADOR DE ESTADO MEJORADO


class StateEncoder:
    """Convierte visión 5x5 + atributos + posición en clave única"""

    def encode(self, vision, gold, map_idx, nivel, agent_pos):
        req = GameConfig.get_gold_requirement(map_idx)
        can_exit = gold >= req

        #  0=Vacío, 1=Pared, 2=Tesoro, 3=Salida
        surroundings = []
        cx, cy = 2, 2  # Centro de la visión

        # Norte, Sur, Este, Oeste
        for dy, dx in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            cell = vision[cy + dy, cx + dx]

            if cell == "#":
                val = 1  # Pared
            elif cell == "T":
                val = 2  # Tesoro
            elif cell == "R":
                val = 3  # Salida
            elif cell in ["X", "E"]:
                val = 4  # Peligro
            else:
                val = 0  # Espacio vacío / Camino seguro

            surroundings.append(val)

        surroundings = tuple(surroundings)
        # -----------------------

        # REMOVIDO: pos_exact para permitir generalización a mapas nuevos
        # El agente debe aprender a navegar basándose en características locales,
        # no en coordenadas absolutas

        # Categorizar nivel en rangos significativos (alineado con requisitos de objetos)
        # 1-15: básico, 16-40: raro, 41-60: épico, 61-100: legendario
        if nivel <= 15:
            rod_tier = 0
        elif nivel <= 40:
            rod_tier = 1
        elif nivel <= 60:
            rod_tier = 2
        else:
            rod_tier = 3

        # Detectar qué hay bajo los pies
        center_cell = vision[2, 2]
        on_exit = 1 if center_cell == "R" else 0

        # Estado simplificado para generalización
        # Solo usa características observables localmente
        return (can_exit, surroundings, rod_tier, on_exit)


# 4. AGENTE CON EXPORTACIÓN


class PersistentAgent:
    def __init__(self, num_maps=12):
        self.q_tables = [{} for _ in range(num_maps)]
        self.encoder = StateEncoder()
        self.pathfinder = PathFinder()  # NUEVO: A* para navegación inteligente

        self.gold = 0
        self.nivel = 1  # NUEVO: Nivel del agente (1-100) para usar objetos
        self.exp = 0  # Experiencia acumulada
        self.exp_multiplicador = 100  # AUMENTADO: 100x para subir mucho más rápido

        # Sistema de inventario
        self.cana_equipada = crear_cana_basica_inicial()  # NUEVO: Caña básica inicial
        self.inventario_canas = []  # Lista de cañas para vender (máximo 5)
        self.inventario_tesoro = None  # Tesoro para vender (solo 1)
        self.max_inventario_canas = 5

        # HIPERPARÁMETROS MEJORADOS
        self.lr = 0.3  # AUMENTADO: Aprender más rápido con rewards positivos
        self.gamma = 0.98  # AUMENTADO: Valorar más las recompensas futuras
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # AJUSTADO: Decay más lento para explorar más

        self.last_positions = []

        # NUEVO: Detección de loops contra paredes
        self.last_wall_actions = []  # Últimas acciones que resultaron en pared
        self.wall_attempts = 0  # Contador de intentos contra paredes

        # NUEVO: Estadísticas de uso de A*
        self.astar_stats = {
            "astar_used": 0,
            "qtable_used": 0,
            "random_used": 0,
        }

        # Estadísticas de entrenamiento
        self.training_stats = {
            "episodes_completed": 0,
            "total_rewards": [],
            "maps_cleared_per_episode": [],
            "epsilon_history": [],
            "objetos_recogidos": 0,
            "oro_por_ventas": 0,
        }

    def _buscar_objetivo_cercano(self, current_map, agent_pos, radio=8):
        """Busca el objetivo más cercano (tesoro, tienda, enemigo) en un radio"""
        agent_y, agent_x = agent_pos
        objetivos = []

        # Buscar en el mapa dentro del radio
        for dy in range(-radio, radio + 1):
            for dx in range(-radio, radio + 1):
                ny, nx = agent_y + dy, agent_x + dx

                # Verificar límites
                if 0 <= ny < current_map.shape[0] and 0 <= nx < current_map.shape[1]:
                    cell = current_map[ny, nx]
                    dist = abs(dy) + abs(dx)  # Manhattan

                    # Prioridades: T (tesoro) > P (tienda si tiene items) > X/E (enemigos)
                    if cell == "T":
                        objetivos.append((dist * 0.5, (ny, nx)))  # Prioridad alta
                    elif cell == "P" and (
                        len(self.inventario_canas) > 0 or self.inventario_tesoro
                    ):
                        objetivos.append((dist * 0.7, (ny, nx)))  # Prioridad media-alta
                    elif cell in ["X", "E"]:
                        objetivos.append((dist * 1.0, (ny, nx)))  # Prioridad media

        # Devolver el más cercano
        if objetivos:
            objetivos.sort(key=lambda x: x[0])
            return objetivos[0][1]

        return None

    def get_action(self, vision, map_idx, agent_pos, current_map=None):
        """
        Decide la acción usando sistema híbrido inteligente: Geometría > A* > Q-Learning

        Jerarquía de decisión:
        1. Escape de salida (si muy cerca sin oro suficiente)
        2. A* forzado a salida (si tiene oro suficiente)
        3. Búsqueda geométrica de objetivos (si necesita oro) - SIEMPRE prioritaria
        4. Q-table con filtrado de paredes
        5. A* como backup (si Q-values ≤ 0.1)
        6. Heurísticas (estado desconocido)

        Args:
            vision: Ventana 5x5 alrededor del agente
            map_idx: Índice del mapa actual
            agent_pos: Posición actual del agente
            current_map: Mapa completo (opcional, para sistema híbrido)
        """
        state = self.encoder.encode(vision, self.gold, map_idx, self.nivel, agent_pos)
        gold_req = GameConfig.get_gold_requirement(map_idx)

        # PRIORIDAD 1: EVITAR la salida si no tiene oro suficiente
        if self.gold < gold_req and current_map is not None:
            exit_positions = np.where((current_map == "R") | (current_map == "G"))
            if len(exit_positions[0]) > 0:
                exit_y, exit_x = exit_positions[0][0], exit_positions[1][0]
                agent_y, agent_x = agent_pos
                dist_to_exit = abs(exit_y - agent_y) + abs(exit_x - agent_x)

                if dist_to_exit <= 2:
                    # MUY CERCA de la salida sin oro - ALEJARSE
                    surroundings = state[1]  # (N, S, E, O)
                    dy = exit_y - agent_y
                    dx = exit_x - agent_x

                    # Movimientos que ALEJAN de la salida
                    escape_moves = []
                    if dy < 0 and surroundings[1] != 1:  # Salida al norte, ir al sur
                        escape_moves.append(1)
                    elif dy > 0 and surroundings[0] != 1:  # Salida al sur, ir al norte
                        escape_moves.append(0)
                    if dx < 0 and surroundings[2] != 1:  # Salida al oeste, ir al este
                        escape_moves.append(2)
                    elif dx > 0 and surroundings[3] != 1:  # Salida al este, ir al oeste
                        escape_moves.append(3)

                    if escape_moves:
                        return np.random.choice(escape_moves)

        # PRIORIDAD ABSOLUTA: Si tiene oro suficiente, IR A LA SALIDA
        # NO buscar tesoros, NO usar Q-table, SOLO A* hacia salida
        if self.gold >= gold_req and current_map is not None:
            astar_action = self.pathfinder.get_next_action(
                current_map, agent_pos, self.gold, gold_req
            )
            if astar_action is not None:
                self.astar_stats["astar_used"] += 1
                return astar_action

            # Si A* falla, usar dirección geométrica hacia salida
            exit_positions = np.where((current_map == "R") | (current_map == "G"))
            if len(exit_positions[0]) > 0:
                exit_y, exit_x = exit_positions[0][0], exit_positions[1][0]
                agent_y, agent_x = agent_pos

                # Calcular dirección hacia la salida
                dy = exit_y - agent_y
                dx = exit_x - agent_x

                # Priorizar movimiento según distancia mayor
                surroundings = state[1]  # (N, S, E, O)
                moves_priority = []

                if abs(dy) > abs(dx):  # Priorizar vertical
                    if dy < 0 and surroundings[0] != 1:  # Norte
                        moves_priority.append(0)
                    elif dy > 0 and surroundings[1] != 1:  # Sur
                        moves_priority.append(1)
                    if dx > 0 and surroundings[2] != 1:  # Este
                        moves_priority.append(2)
                    elif dx < 0 and surroundings[3] != 1:  # Oeste
                        moves_priority.append(3)
                else:  # Priorizar horizontal
                    if dx > 0 and surroundings[2] != 1:  # Este
                        moves_priority.append(2)
                    elif dx < 0 and surroundings[3] != 1:  # Oeste
                        moves_priority.append(3)
                    if dy < 0 and surroundings[0] != 1:  # Norte
                        moves_priority.append(0)
                    elif dy > 0 and surroundings[1] != 1:  # Sur
                        moves_priority.append(1)

                # Agregar movimientos restantes no-pared
                for i in range(4):
                    if i not in moves_priority and surroundings[i] != 1:
                        moves_priority.append(i)

                if moves_priority:
                    self.astar_stats["qtable_used"] += 1
                    return moves_priority[0]

            # Último recurso: cualquier dirección válida
            surroundings = state[1]
            valid = [i for i in range(4) if surroundings[i] != 1]
            if valid:
                self.astar_stats["random_used"] += 1
                return np.random.choice(valid)

            # TERMINAR AQUÍ - No continuar si tiene oro suficiente
            self.astar_stats["random_used"] += 1
            return np.random.randint(4)

        # A partir de aquí, solo ejecutar si NO tiene oro suficiente
        if map_idx >= len(self.q_tables):
            return np.random.randint(4)

        table = self.q_tables[map_idx]

        # CRÍTICO: Detectar si está atascado contra paredes
        if len(self.last_wall_actions) >= 3:
            # Si las últimas 3 acciones fueron contra paredes, usar A* o explorar
            if current_map is not None:
                astar_action = self.pathfinder.get_next_action(
                    current_map, agent_pos, self.gold, gold_req
                )
                if astar_action is not None:
                    self.last_wall_actions = []  # Reset
                    self.wall_attempts = 0
                    return astar_action

            # Si A* falla, explorar aleatoriamente (evitar direcciones previas)
            surroundings = state[1]
            valid_moves = [i for i in range(4) if surroundings[i] != 1]
            # Filtrar acciones que causaron paredes recientemente
            safe_moves = [
                a for a in valid_moves if a not in self.last_wall_actions[-3:]
            ]
            if safe_moves:
                self.last_wall_actions = []  # Reset
                self.wall_attempts = 0
                return np.random.choice(safe_moves)
            elif valid_moves:
                self.last_wall_actions = []  # Reset
                self.wall_attempts = 0
                return np.random.choice(valid_moves)

        # PRIORIDAD 3: SIEMPRE buscar objetivos geométricamente cuando necesita oro
        # (ANTES de consultar Q-table para evitar interferencia)
        if self.gold < gold_req and current_map is not None:
            # Radio 8 por defecto, 12 cuando conoce el estado
            radio = 12 if state in table else 8
            objetivo_pos = self._buscar_objetivo_cercano(current_map, agent_pos, radio)
            if objetivo_pos is not None:
                # Calcular dirección geométrica hacia el objetivo
                obj_y, obj_x = objetivo_pos
                agent_y, agent_x = agent_pos
                dy = obj_y - agent_y
                dx = obj_x - agent_x

                surroundings = state[1]  # (N, S, E, O)
                moves_priority = []

                # Priorizar movimiento según distancia mayor (geometría directa)
                if abs(dy) > abs(dx):
                    if dy < 0 and surroundings[0] != 1:  # Norte
                        moves_priority.append(0)
                    elif dy > 0 and surroundings[1] != 1:  # Sur
                        moves_priority.append(1)
                    if dx > 0 and surroundings[2] != 1:  # Este
                        moves_priority.append(2)
                    elif dx < 0 and surroundings[3] != 1:  # Oeste
                        moves_priority.append(3)
                else:
                    if dx > 0 and surroundings[2] != 1:  # Este
                        moves_priority.append(2)
                    elif dx < 0 and surroundings[3] != 1:  # Oeste
                        moves_priority.append(3)
                    if dy < 0 and surroundings[0] != 1:  # Norte
                        moves_priority.append(0)
                    elif dy > 0 and surroundings[1] != 1:  # Sur
                        moves_priority.append(1)

                # Agregar movimientos restantes
                for i in range(4):
                    if i not in moves_priority and surroundings[i] != 1:
                        moves_priority.append(i)

                if moves_priority:
                    # SIEMPRE usar la dirección geométrica más directa al objetivo
                    # No dejar que Q-values interfieran y causen oscilación
                    return moves_priority[0]

        # Si estado desconocido, usar heurísticas mejoradas
        if state not in table:
            table[state] = np.zeros(4)

            # Buscar objetivos activamente
            if current_map is not None and self.gold < gold_req:
                objetivo_pos = self._buscar_objetivo_cercano(current_map, agent_pos)
                if objetivo_pos is not None:
                    obj_y, obj_x = objetivo_pos
                    agent_y, agent_x = agent_pos
                    dy = obj_y - agent_y
                    dx = obj_x - agent_x

                    surroundings = state[1]
                    moves_priority = []

                    if abs(dy) > abs(dx):
                        if dy < 0 and surroundings[0] != 1:
                            moves_priority.append(0)
                        elif dy > 0 and surroundings[1] != 1:
                            moves_priority.append(1)
                        if dx > 0 and surroundings[2] != 1:
                            moves_priority.append(2)
                        elif dx < 0 and surroundings[3] != 1:
                            moves_priority.append(3)
                    else:
                        if dx > 0 and surroundings[2] != 1:
                            moves_priority.append(2)
                        elif dx < 0 and surroundings[3] != 1:
                            moves_priority.append(3)
                        if dy < 0 and surroundings[0] != 1:
                            moves_priority.append(0)
                        elif dy > 0 and surroundings[1] != 1:
                            moves_priority.append(1)

                    for i in range(4):
                        if i not in moves_priority and surroundings[i] != 1:
                            moves_priority.append(i)

                    if moves_priority:
                        return moves_priority[0]

                # Si no hay objetivos, usar A*
                astar_action = self.pathfinder.get_next_action(
                    current_map, agent_pos, self.gold, gold_req
                )
                if astar_action is not None:
                    self.astar_stats["astar_used"] += 1
                    return astar_action

            # Heurística mejorada: priorizar tesoros > enemigos > vacíos
            surroundings = state[1]
            on_exit = state[3]

            if on_exit and state[0]:  # Estamos en salida con oro
                return np.random.randint(4)

            moves_tesoro = []
            moves_enemigo = []
            moves_vacio = []

            for i, val in enumerate(surroundings):
                if val == 2:  # Tesoro
                    moves_tesoro.append(i)
                elif val == 4:  # Enemigo
                    moves_enemigo.append(i)
                elif val == 0:  # Vacío
                    moves_vacio.append(i)

            if moves_tesoro:
                return np.random.choice(moves_tesoro)
            if moves_enemigo and self.gold < gold_req:
                return np.random.choice(moves_enemigo)
            if moves_vacio:
                return np.random.choice(moves_vacio)

            # Todo bloqueado, elegir no-pared
            moves = [i for i, val in enumerate(surroundings) if val != 1]
            if moves:
                return np.random.choice(moves)

            return np.random.randint(4)

        # Política epsilon-greedy
        if np.random.rand() < self.epsilon:
            # 50% A*, 50% aleatorio durante exploración
            if current_map is not None and np.random.rand() < 0.5:
                astar_action = self.pathfinder.get_next_action(
                    current_map, agent_pos, self.gold, gold_req
                )
                if astar_action is not None:
                    self.astar_stats["astar_used"] += 1
                    return astar_action

            # Exploración: elegir acción aleatoria (preferir no-paredes)
            surroundings = state[1]
            moves = [i for i, val in enumerate(surroundings) if val != 1]
            if moves:
                self.astar_stats["random_used"] += 1
                return np.random.choice(moves)
            return np.random.randint(4)

        # PRIORIDAD 4: Usar Q-table con FILTRADO DE PAREDES
        surroundings = state[1]  # (N, S, E, O)
        valid_actions = [i for i in range(4) if surroundings[i] != 1]  # No-paredes

        if not valid_actions:
            # Todas son paredes (no debería pasar)
            return np.random.randint(4)

        # Filtrar Q-values solo de acciones válidas (no-paredes)
        valid_q_values = [(i, table[state][i]) for i in valid_actions]
        max_val = max(v for _, v in valid_q_values)

        # PRIORIDAD 5: Si Q-values muy bajos, usar A* como backup
        if max_val <= 0.1 and current_map is not None:
            astar_action = self.pathfinder.get_next_action(
                current_map, agent_pos, self.gold, gold_req
            )
            if astar_action is not None:
                self.astar_stats["astar_used"] += 1
                return astar_action

        # Elegir entre las mejores acciones válidas (no-paredes)
        self.astar_stats["qtable_used"] += 1
        best_actions = [i for i, v in valid_q_values if v == max_val]
        return np.random.choice(best_actions)

    def learn(
        self,
        vision,
        action,
        reward,
        next_vision,
        map_idx,
        done,
        agent_pos,
        next_agent_pos,
    ):
        state = self.encoder.encode(vision, self.gold, map_idx, self.nivel, agent_pos)
        next_state = self.encoder.encode(
            next_vision, self.gold, map_idx, self.nivel, next_agent_pos
        )

        table = self.q_tables[map_idx]
        if state not in table:
            table[state] = np.zeros(4)
        if next_state not in table:
            table[next_state] = np.zeros(4)

        target = reward
        if not done:
            target += self.gamma * np.max(table[next_state])

        predict = table[state][action]
        table[state][action] += self.lr * (target - predict)

        # Detector de ciclos mejorado (4 pasos)
        self.last_positions.append(agent_pos)
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)

        # Detectar ciclos después de actualizar posiciones
        if len(self.last_positions) >= 4:
            # Detectar A-B-A-B (oscilación)
            if (
                self.last_positions[-1] == self.last_positions[-3]
                and self.last_positions[-2] == self.last_positions[-4]
            ):
                # Eliminar estado problemático para forzar re-aprendizaje
                if state in table:
                    del table[state]
                self.last_positions = []

            # Detectar 4+ movimientos en el mismo lugar (stuck)
            if len(set(self.last_positions[-4:])) == 1:
                if state in table:
                    del table[state]
                self.last_positions = []

    def exp_para_siguiente_nivel(self, nivel_actual):
        """Calcula experiencia necesaria para subir de nivel (escala suave)"""
        # Fórmula: 80 * nivel^1.2 (antes: 100 * nivel^1.5)
        # Nivel 2: 80 * 2^1.2 = 184 exp (antes: 283)
        # Nivel 3: 80 * 3^1.2 = 280 exp (antes: 520)
        return int(80 * (nivel_actual**1.2))

    def agregar_experiencia(self, exp_ganada):
        """Agrega experiencia y sube de nivel si es necesario"""
        exp_real = exp_ganada * self.exp_multiplicador
        self.exp += exp_real

        niveles_subidos = 0
        while (
            self.exp >= self.exp_para_siguiente_nivel(self.nivel) and self.nivel < 100
        ):
            self.exp -= self.exp_para_siguiente_nivel(self.nivel)
            self.nivel += 1
            niveles_subidos += 1

        return niveles_subidos, exp_real

    def gestionar_objeto(self, objeto):
        """Gestiona un objeto encontrado según su tipo"""
        tipo = objeto.tipo_objeto

        # Basura: descartar inmediatamente
        if tipo in ["anillas_sixpack", "botella", "lata", "llanta"]:
            return {"accion": "descartado", "oro_ganado": 0}

        # Tesoro: guardar en inventario
        if tipo == "tesoro":
            if self.inventario_tesoro is None:
                self.inventario_tesoro = objeto
                return {"accion": "tesoro_guardado", "oro_ganado": 0}
            else:
                # Ya hay un tesoro, vender el de menor valor
                if objeto.precio_venta > self.inventario_tesoro.precio_venta:
                    oro_ganado = self.inventario_tesoro.precio_venta
                    self.inventario_tesoro = objeto
                    return {"accion": "tesoro_reemplazado", "oro_ganado": oro_ganado}
                else:
                    return {
                        "accion": "tesoro_vendido_directo",
                        "oro_ganado": objeto.precio_venta,
                    }

        # Caña: comparar con equipada y agregar al inventario
        if tipo in ["cana_basica", "cana_rara", "cana_epica", "cana_legendaria"]:
            oro_ganado = 0

            # Validar nivel requerido
            if self.nivel < objeto.nivel_requerido:
                print(
                    f"Caña requiere nivel {objeto.nivel_requerido} (Actual: {self.nivel})"
                )
                # Intentar guardar para cuando suba de nivel
                if len(self.inventario_canas) < self.max_inventario_canas:
                    self.inventario_canas.append(objeto)
                    print("Guardada en inventario para cuando subas de nivel")
                    return {"accion": "cana_guardada_nivel_bajo", "oro_ganado": 0}
                else:
                    print("Inventario lleno, caña descartada")
                    return {"accion": "cana_descartada_nivel_bajo", "oro_ganado": 0}

            # Si no tiene caña equipada
            if self.cana_equipada is None:
                self.cana_equipada = objeto
                print(f"⚡ Caña equipada: {tipo}")
                return {"accion": "cana_equipada", "oro_ganado": 0}

            # Comparar con caña equipada (prioridad: rareza > precio)
            def es_mejor(nueva, actual):
                if nueva.rareza > actual.rareza:
                    return True
                elif nueva.rareza == actual.rareza:
                    return nueva.precio_venta > actual.precio_venta
                return False

            if es_mejor(objeto, self.cana_equipada):
                # La nueva es mejor: mover equipada al inventario y equipar nueva
                cana_antigua = self.cana_equipada
                self.cana_equipada = objeto

                # Intentar guardar la caña antigua en inventario
                if len(self.inventario_canas) < self.max_inventario_canas:
                    self.inventario_canas.append(cana_antigua)
                    return {"accion": "cana_mejorada_guardada", "oro_ganado": 0}
                else:
                    # Inventario lleno: vender la caña de menor valor
                    # Agregar la antigua al inventario temporalmente para comparar
                    todas_canas = self.inventario_canas + [cana_antigua]
                    # Ordenar por valor (rareza primero, luego precio)
                    todas_canas.sort(key=lambda c: (c.rareza, c.precio_venta))
                    # Vender la peor y quedarse con las mejores 5
                    cana_a_vender = todas_canas[0]
                    self.inventario_canas = todas_canas[1:]
                    oro_ganado = cana_a_vender.precio_venta
                    return {
                        "accion": "cana_mejorada_vendio_peor",
                        "oro_ganado": oro_ganado,
                    }
            else:
                # La equipada es mejor: intentar guardar o vender la nueva
                if len(self.inventario_canas) < self.max_inventario_canas:
                    self.inventario_canas.append(objeto)
                    return {"accion": "cana_guardada", "oro_ganado": 0}
                else:
                    # Inventario lleno: comparar con la peor del inventario
                    todas_canas = self.inventario_canas + [objeto]
                    todas_canas.sort(key=lambda c: (c.rareza, c.precio_venta))
                    # Vender la peor
                    cana_a_vender = todas_canas[0]
                    self.inventario_canas = todas_canas[1:]
                    oro_ganado = cana_a_vender.precio_venta
                    return {
                        "accion": "cana_reemplazo_inventario",
                        "oro_ganado": oro_ganado,
                    }

        return {"accion": "desconocido", "oro_ganado": 0}

    def vender_inventario(self):
        """Vende todo el inventario en la tienda"""
        oro_total = 0

        # Vender todas las cañas del inventario
        for cana in self.inventario_canas:
            oro_total += cana.precio_venta
        if len(self.inventario_canas) > 0:
            self.inventario_canas = []

        # Vender tesoro si hay
        if self.inventario_tesoro is not None:
            oro_total += self.inventario_tesoro.precio_venta
            self.inventario_tesoro = None

        if oro_total > 0:
            self.training_stats["oro_por_ventas"] += oro_total

        return oro_total

    def tiene_mejor_cana(self):
        """Verifica si el agente tiene una caña de alta calidad (épica o legendaria)"""
        if self.cana_equipada:
            return self.cana_equipada.tipo_objeto in ["cana_epica", "cana_legendaria"]
        return False

    #  FUNCIONES DE EXPORTACIÓN

    def save_agent(self, filepath="agente_entrenado.pkl"):
        """Guarda el agente completo (Q-tables + parámetros)"""
        agent_data = {
            "q_tables": self.q_tables,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "training_stats": self.training_stats,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(agent_data, f)

        print(f" Agente guardado en: {filepath}")
        print(f"   - Q-Tables: {len(self.q_tables)} mapas")
        print(f"   - Estados totales: {sum(len(table) for table in self.q_tables)}")

    def load_agent(self, filepath="agente_entrenado.pkl"):
        """Carga un agente previamente entrenado"""
        with open(filepath, "rb") as f:
            agent_data = pickle.load(f)

        self.q_tables = agent_data["q_tables"]
        self.epsilon = agent_data["epsilon"]
        self.lr = agent_data["lr"]
        self.gamma = agent_data["gamma"]
        self.training_stats = agent_data.get("training_stats", {})

        print(f"Agente cargado desde: {filepath}")
        print(f"   - Entrenado el: {agent_data.get('timestamp', 'desconocido')}")
        print(f"   - Estados aprendidos: {sum(len(table) for table in self.q_tables)}")

    def export_qtable_readable(self, map_idx=0, filepath="qtable_mapa_0.txt", top_n=50):
        """Exporta Q-table legible en texto (top N estados)"""
        if map_idx >= len(self.q_tables):
            print(f" Mapa {map_idx} no existe")
            return

        table = self.q_tables[map_idx]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Q-TABLE PARA MAPA {map_idx}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total de estados: {len(table)}\n")
            f.write(f"Acciones: 0=Norte, 1=Sur, 2=Este, 3=Oeste\n\n")

            # Ordenar por valor máximo de Q
            sorted_states = sorted(
                table.items(), key=lambda x: np.max(x[1]), reverse=True
            )

            f.write(f"TOP {min(top_n, len(sorted_states))} ESTADOS MÁS VALIOSOS:\n")
            f.write("-" * 80 + "\n\n")

            for i, (state, q_values) in enumerate(sorted_states[:top_n]):
                f.write(f"Estado #{i+1}:\n")
                f.write(f"  Descripción: {state}\n")
                f.write(
                    f"  Q-Valores: N={q_values[0]:.2f}, S={q_values[1]:.2f}, "
                    f"E={q_values[2]:.2f}, O={q_values[3]:.2f}\n"
                )
                f.write(
                    f"  Mejor acción: {['Norte', 'Sur', 'Este', 'Oeste'][np.argmax(q_values)]}\n"
                )
                f.write("\n")

        print(f" Q-Table exportada a: {filepath}")

    def export_qtable_json(self, filepath="qtables_completas.json"):
        """Exporta todas las Q-tables en formato JSON"""
        # Convertir q_tables a formato serializable
        serializable_tables = []
        for table in self.q_tables:
            serializable_table = {}
            for state, q_values in table.items():
                # Convertir tupla a string para usar como key en JSON
                state_key = str(state)
                serializable_table[state_key] = q_values.tolist()
            serializable_tables.append(serializable_table)

        data = {
            "q_tables": serializable_tables,
            "metadata": {
                "num_maps": len(self.q_tables),
                "total_states": sum(len(table) for table in self.q_tables),
                "epsilon": self.epsilon,
                "timestamp": datetime.now().isoformat(),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Q-Tables JSON exportadas a: {filepath}")


# 5. LOOP PRINCIPAL

if __name__ == "__main__":
    # Usar mapas fijos de la carpeta 'mapa' para entrenamiento inicial
    env = MapEnvironment(maps_folder="mapa", use_random=False)
    agent = PersistentAgent(num_maps=12)  # 12 mapas

    EPISODES = 5000
    print(f"\n{'='*60}")
    print(f"   INICIANDO ENTRENAMIENTO ({EPISODES} Episodios)")
    print(f"   Sistema de Oro: Activo (requisitos por nivel)")
    print(f"   Combates: logica_combate")
    print(f"   Objetos: logica_objetos")
    print(f"   Objetivo: Llegar al mapa 12 con buena caña")
    print(f"{'='*60}\n")

    try:
        for episode in range(EPISODES):
            vision = env.reset(keep_progress=False)
            # El oro, nivel, experiencia (exp) e inventario persisten entre episodios
            # Solo se resetea al inicio del primer episodio
            if episode == 0:
                agent.gold = 0
            agent.last_positions = []  # Reset historial de posiciones

            total_reward = 0
            maps_cleared = 0

            while True:
                current_map = env.current_map_idx
                old_pos = env.agent_pos

                # 1. Decidir (pasar mapa completo para A*)
                action = agent.get_action(
                    vision, current_map, env.agent_pos, env.maps[env.current_map_idx]
                )

                # 2. Actuar (pasar nivel y rareza de caña)
                cana_rareza = agent.cana_equipada.rareza if agent.cana_equipada else 1
                next_vision, reward, done, info = env.step(
                    action, agent.gold, agent.nivel, cana_rareza
                )

                # 3. Actualizar Stats RPG
                if "gold_gain" in info:
                    oro_antes = agent.gold
                    agent.gold += info["gold_gain"]
                    # Prevenir oro negativo
                    if agent.gold < 0:
                        print(
                            f"DEBUG: Oro negativo detectado! {oro_antes} + {info['gold_gain']} = {agent.gold}"
                        )
                        agent.gold = 0

                # Ganar experiencia de combates
                if "exp_gain" in info:
                    niveles_subidos, exp_real = agent.agregar_experiencia(
                        info["exp_gain"]
                    )
                    if niveles_subidos > 0:
                        print(
                            f"¡Subiste {niveles_subidos} nivel(es)! Ahora eres nivel {agent.nivel} (permanente)"
                        )

                # 4. Aprender
                level_finished = info.get("level_complete", False)
                agent.learn(
                    vision,
                    action,
                    reward,
                    next_vision,
                    current_map,
                    done,
                    old_pos,
                    env.agent_pos,
                )

                # Gestionar objeto encontrado
                if "objeto_encontrado" in info and info.get("objeto_recogido"):
                    resultado = agent.gestionar_objeto(info["objeto_encontrado"])
                    if resultado["oro_ganado"] > 0:
                        agent.gold += resultado["oro_ganado"]
                    agent.training_stats["objetos_recogidos"] += 1

                # Vender inventario en tienda
                if info.get("venta_realizada"):
                    oro_vendido = agent.vender_inventario()
                    agent.gold += oro_vendido

                # Upgrade en tienda
                vision = next_vision
                total_reward += reward

                if done:
                    # Verificar si completó el nivel (level_complete)
                    if info.get("level_complete"):
                        maps_cleared += 1
                        mapa_completado = env.current_map_idx

                        # Verificar victoria total
                        if mapa_completado == 11 and agent.tiene_mejor_cana():
                            print(f"\n{'='*60}")
                            print(f"¡VICTORIA TOTAL!")
                            print(
                                f"  Completaste el mapa 12 con caña: {agent.cana_equipada.tipo_objeto}"
                            )
                            print(f"  Oro final: {agent.gold}")
                            print(f"  Nivel: {agent.nivel}")
                            print(f"{'='*60}\n")
                            # Continuar entrenando (volver al inicio)
                            env.current_map_idx = 0
                            vision = env.reset(keep_progress=False)
                            # Resetear oro y nivel para nueva partida
                            agent.gold = 0
                            agent.nivel = 1
                            agent.exp = 0
                            agent.cana_equipada = crear_cana_basica_inicial()
                            agent.inventario_canas = []
                            agent.inventario_tesoro = None
                        else:
                            # Avanzar al siguiente mapa
                            env.current_map_idx += 1

                            if env.current_map_idx >= len(env.maps):
                                # Completó todos los mapas disponibles, reiniciar
                                print(
                                    f"\n¡Completaste los {len(env.maps)} mapas! Reiniciando..."
                                )
                                env.current_map_idx = 0
                                vision = env.reset(keep_progress=False)
                                # Resetear para nueva partida
                                agent.gold = 0
                                agent.nivel = 1
                                agent.exp = 0
                                agent.cana_equipada = crear_cana_basica_inicial()
                                agent.inventario_canas = []
                                agent.inventario_tesoro = None
                            else:
                                # Continuar al siguiente mapa manteniendo progreso
                                vision = env.reset(keep_progress=True)
                                agent.last_positions = []

                        done = False  # Continuar episodio
                    else:
                        # Timeout o game over
                        break

            # Actualizar estadísticas
            agent.training_stats["total_rewards"].append(total_reward)
            agent.training_stats["maps_cleared_per_episode"].append(maps_cleared)
            agent.training_stats["epsilon_history"].append(agent.epsilon)
            agent.training_stats["episodes_completed"] = episode + 1

            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

            if episode % 50 == 0:
                avg_reward = (
                    np.mean(agent.training_stats["total_rewards"][-50:])
                    if len(agent.training_stats["total_rewards"]) >= 50
                    else total_reward
                )
                exp_sig = agent.exp_para_siguiente_nivel(agent.nivel)
                cana_nombre = (
                    agent.cana_equipada.tipo_objeto
                    if agent.cana_equipada
                    else "ninguna"
                )
                print(
                    f"Ep: {episode:4d} | ε: {agent.epsilon:.3f} | Nivel: {agent.nivel} (EXP: {agent.exp}/{exp_sig}) | Caña: {cana_nombre} | "
                    f"Mapas: {maps_cleared} | Mapa Actual: {env.current_map_idx} | Oro: {agent.gold:4d} | Reward: {total_reward:6.0f} | Avg50: {avg_reward:6.0f}"
                )

    except KeyboardInterrupt:
        print("\n  Entrenamiento interrumpido por usuario.")

    #  EXPORTAR AGENTE Y Q-TABLES
    print("\n" + "=" * 60)
    print("   GUARDANDO RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 60 + "\n")

    # Exportar todas las Q-tables en JSON
    agent.export_qtable_json("qtables_completas.json")

    # Mostrar estadísticas de uso de A*
    total_decisions = sum(agent.astar_stats.values())
    if total_decisions > 0:
        print(f"\nEstadísticas de Navegación:")
        print(
            f"   A* usado: {agent.astar_stats['astar_used']} veces ({agent.astar_stats['astar_used']/total_decisions*100:.1f}%)"
        )
        print(
            f"   Q-Table usado: {agent.astar_stats['qtable_used']} veces ({agent.astar_stats['qtable_used']/total_decisions*100:.1f}%)"
        )
        print(
            f"   Random usado: {agent.astar_stats['random_used']} veces ({agent.astar_stats['random_used']/total_decisions*100:.1f}%)"
        )

    print("\nEntrenamiento completado y guardado.\n")
