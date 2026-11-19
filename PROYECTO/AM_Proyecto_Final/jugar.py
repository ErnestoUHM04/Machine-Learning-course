import numpy as np
import pickle
import time
import os
import sys
import glob
from logica_combate.logica_combate import SistemaCombate
from logica_objetos.logica_objetos import ObjetoLogico
import random
import modulo_dia_noche as motor_clima
import modulo4_hmm as ia_enemigos

# ==========================================
# IMPORTAR SOLO EL CODIFICADOR DE ESTADO
# ==========================================
from agente1 import (
    StateEncoder,
    GameConfig,
    crear_cana_basica_inicial,
    PathFinder,
    OBJETOS_CONFIG,
)
from generador_aleatorio import GeneradorEscenariosAleatorios
from logica_combate.logica_combate import SistemaCombate
from logica_objetos.logica_objetos import ObjetoLogico
import random

# ==========================================
# UTILIDADES
# ==========================================


def generar_mapas_si_no_existen(carpeta, num_mapas=12, seed=None):
    """Genera mapas aleatorios si la carpeta no existe o está vacía"""
    if not os.path.exists(carpeta):
        print(f"\nCarpeta '{carpeta}' no encontrada")
        print(f"Generando {num_mapas} mapas aleatorios...\n")

        os.makedirs(carpeta)
        generador = GeneradorEscenariosAleatorios(seed=seed)

        for i in range(num_mapas):
            dificultad = i + 1

            tipo = "abierto"

            print(f"  Generando mapa {i+1}/{num_mapas} (tipo: {tipo})...", end=" ")
            mapa = generador.generar_escenario_aleatorio(
                dificultad=dificultad, tipo_mapa=tipo
            )

            if mapa:
                archivo = os.path.join(carpeta, f"escenario_{i+1:02d}.txt")
                with open(archivo, "w", encoding="utf-8") as f:
                    for fila in mapa:
                        f.write("".join(fila) + "\n")
                print("✔")
            else:
                print("x")

        print(f"\nMapas generados en '{carpeta}'\n")
        return True

    # Verificar si tiene mapas
    mapas = glob.glob(os.path.join(carpeta, "*.txt"))
    if not mapas:
        print(f"\nCarpeta '{carpeta}' existe pero está vacía")
        respuesta = (
            input(f"¿Generar {num_mapas} mapas aleatorios? (S/n): ").strip().lower()
        )

        if respuesta in ["", "s", "si", "sí", "y", "yes"]:
            return generar_mapas_si_no_existen(carpeta, num_mapas, seed=None)
        else:
            return False

    return True


# ==========================================
# ENTORNO SIMPLIFICADO (sin combates ni objetos)
# ==========================================


class SimpleMapEnvironment:
    """Versión simplificada del entorno para visualización"""

    def __init__(self, maps_folder="mapa"):
        self.maps_folder = maps_folder
        self.map_templates = self._load_maps()
        self.maps = []
        self.current_map_idx = 0
        self.agent_pos = (1, 1)
        self.max_steps = 1000
        self.steps = 0

        # --- NUEVO: ESTADO ATMOSFÉRICO E IA ---
        self.estado_furia = 1  # 1: Normal, 3: Enojado, 6: Encabronado
        self.clima_actual_nombre = "Soleado"
        self.es_dia = True  # Rastrear día/noche localmente
        motor_clima.es_dia = True  # Inicializar variable global del módulo
        # Inicializar sistema de combate
        self.combate = SistemaCombate(
            model_path="logica_combate/modelo_rf.pkl",
            dataset_path="logica_combate/dataset_peces_enemigos.csv",
        )

    def _load_maps(self):
        if not os.path.exists(self.maps_folder):
            raise ValueError(f"Carpeta '{self.maps_folder}' no encontrada")

        files = sorted(glob.glob(os.path.join(self.maps_folder, "*.txt")))
        loaded_maps = []

        print(f"Cargando mapas desde {self.maps_folder}...")
        SIZE = GameConfig.MAP_SIZE

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [
                    line.replace("\n", "").replace("\r", "")
                    for line in f.readlines()
                    if line.strip()
                ]

            # Ajustar tamaño
            if len(lines) < SIZE:
                lines += ["#" * SIZE] * (SIZE - len(lines))
            elif len(lines) > SIZE:
                lines = lines[:SIZE]

            grid_list = []
            for line in lines:
                l = list(line.ljust(SIZE, "#"))[:SIZE]
                grid_list.append(l)

            loaded_maps.append(np.array(grid_list))
            print(f" -> {os.path.basename(file_path)} cargado ({SIZE}x{SIZE}).")

        if not loaded_maps:
            raise ValueError("No se encontraron mapas válidos.")

        return loaded_maps

    def reset(self, keep_progress=False):
        if not keep_progress:
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]

        if self.current_map_idx >= len(self.maps):
            self.current_map_idx = 0
            self.maps = [m.copy() for m in self.map_templates]
        else:
            # CRÍTICO: Regenerar el mapa actual desde el template
            # Esto restaura enemigos (E, X), tesoros (T), etc. cuando avanzas de nivel
            if keep_progress:
                self.maps[self.current_map_idx] = self.map_templates[
                    self.current_map_idx
                ].copy()

        current_map = self.maps[self.current_map_idx]
        start_positions = np.where(current_map == "S")

        if len(start_positions[0]) > 0:
            # Nota: en agente1.py es (y, x) no (x, y)
            self.agent_pos = (start_positions[0][0], start_positions[1][0])

        self.steps = 0
        return self.get_vision()

    def _respawn_enemies(self):
        """Respawnea enemigos en celdas vacías para evitar quedarse sin recursos"""
        current_map = self.maps[self.current_map_idx]

        # Contar enemigos actuales
        current_enemies = np.sum((current_map == "X") | (current_map == "E"))

        # Solo respawnear si hay menos de 5 enemigos
        if current_enemies < 5:
            # Buscar celdas vacías (no adyacentes al agente)
            agent_y, agent_x = self.agent_pos
            empty_cells = []

            for y in range(1, GameConfig.MAP_SIZE - 1):
                for x in range(1, GameConfig.MAP_SIZE - 1):
                    # Verificar que sea espacio vacío
                    if current_map[y, x] in [" ", "'"]:
                        # No respawnear cerca del agente (radio de 3 celdas)
                        dist = abs(y - agent_y) + abs(x - agent_x)
                        if dist > 3:
                            empty_cells.append((y, x))

            # Respawnear 1-2 enemigos
            if empty_cells:
                num_spawns = min(2, len(empty_cells), 8 - current_enemies)
                spawn_positions = np.random.choice(
                    len(empty_cells), size=num_spawns, replace=False
                )

                for idx in spawn_positions:
                    y, x = empty_cells[idx]
                    # 70% X (débil), 30% E (fuerte)
                    enemy_type = "X" if np.random.rand() < 0.7 else "E"
                    current_map[y, x] = enemy_type

    def get_vision(self):
        """Extrae ventana 5x5"""
        y, x = self.agent_pos
        current_map = self.maps[self.current_map_idx]

        padded = np.pad(current_map, 2, constant_values="#")
        py, px = y + 2, x + 2

        return padded[py - 2 : py + 3, px - 2 : px + 3]

    def step(self, action, agent_gold, agent_nivel, agent_cana_rareza):
        """
        agent_nivel: nivel del agente (1-100) - no usado en jugar.py simplificado
        agent_cana_rareza: rareza de la caña equipada (1-5) para calcular bonos
        """
        self.steps += 1
        # Actualización de Clima e IA (cada 4 pasos)
        if self.steps % 4 == 0:
            codigo_clima, visibilidad = motor_clima.ciclo()
            self.es_dia = motor_clima.es_dia  # Guardar estado de día/noche
            
            # Mapear código de clima a nombre
            clima_nombres = {
                0: "Soleado",
                1: "Lluvia",
                2: "Tormenta",
                3: "Templado",
                4: "Lluvia",
                5: "Tormenta"
            }
            self.clima_actual_nombre = clima_nombres.get(codigo_clima, "Soleado")
            
            # Predecir furia de enemigos según clima
            clima_hmm = (
                1 if codigo_clima in [1, 4] else 2 if codigo_clima in [2, 5] else 0
            )
            resultado_hmm = ia_enemigos.predecir_tipo_enemigos([clima_hmm])
            self.estado_furia = resultado_hmm[-1]

            # NUEVO: Respawnear enemigos cada 12 pasos (3 ciclos de clima)
            if self.steps % 12 == 0:
                self._respawn_enemies()

        y, x = self.agent_pos

        # Mismo orden que en agente1.py: Norte, Sur, Este, Oeste
        moves = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        dy, dx = moves[action]
        ny, nx = y + dy, x + dx

        current_map = self.maps[self.current_map_idx]
        limit = GameConfig.MAP_SIZE

        if not (0 <= ny < limit and 0 <= nx < limit):
            ny, nx = y, x

        cell = current_map[ny, nx]
        reward = 0.5  # Base: +0.5 por movimiento válido (incentiva exploración)
        done = False
        info = {}

        # Recompensa por acercarse a la salida si tiene oro
        req = GameConfig.get_gold_requirement(self.current_map_idx)
        if agent_gold >= req:
            exit_positions = np.where((current_map == "R") | (current_map == "G"))
            if len(exit_positions[0]) > 0:
                min_dist = float("inf")
                for ey, ex in zip(exit_positions[0], exit_positions[1]):
                    dist = abs(ny - ey) + abs(nx - ex)
                    min_dist = min(min_dist, dist)
                if min_dist < 10:
                    reward += (10 - min_dist) * 5  # Aumentado de *2 a *5

        if cell == GameConfig.WALL:
            reward = -3  # Pared: -3 (moderado, 6x el movimiento válido)
            ny, nx = y, x

        elif cell == GameConfig.TREASURE:
            # Solo procesar si nos estamos moviendo HACIA el tesoro (no si ya estamos sobre él)
            old_cell = current_map[y, x]
            if old_cell != GameConfig.TREASURE:
                # Sistema completo de generación de objetos
                # 1. Generar objeto aleatorio usando el modelo
                config_elegida = random.choice(list(OBJETOS_CONFIG.values()))
                objeto = ObjetoLogico(
                    config_elegida, usar_rareza_ponderada=True, predecir_automatico=True
                )

                info["objeto_encontrado"] = objeto

                # 2. Verificar si el agente tiene nivel suficiente
                if agent_nivel < objeto.nivel_requerido:
                    reward = -10  # Penalización por nivel insuficiente
                    info["objeto_rechazado"] = "nivel_insuficiente"
                    info["objeto_nombre"] = objeto.tipo_objeto
                    info["nivel_requerido"] = objeto.nivel_requerido
                    # NO limpiar casilla - el tesoro permanece visible
                    ny, nx = y, x  # No moverse sobre el tesoro
                else:
                    # Recompensa basada en valor estratégico
                    reward = objeto.valor_estrategico * 5
                    info["objeto_recogido"] = True
                    info["objeto_nombre"] = objeto.tipo_objeto
                    info["objeto_rareza"] = objeto.rareza
                    info["objeto_valor"] = objeto.precio_venta
                    info["objeto_tipo"] = objeto.tipo_objeto
                    # Limpiar casilla solo si se recogió exitosamente
                    current_map[ny, nx] = GameConfig.EMPTY

        elif cell in [GameConfig.ENEMY_X, GameConfig.ENEMY_E]:
            # Solo procesar combate si nos estamos moviendo HACIA el enemigo (no si ya estamos sobre él)
            old_cell = current_map[y, x]
            if old_cell not in [GameConfig.ENEMY_X, GameConfig.ENEMY_E]:
                # Sistema completo de combate

                # 1. Generar enemigo base
                pez = self.combate.generar_enemigo_por_zona(self.current_map_idx)

                # --- NUEVA LÓGICA: Modificar enemigo según HMM ---
                multiplicador = 1.0
                if self.estado_furia == 3:  # Enojado
                    multiplicador = 1.3
                elif self.estado_furia == 6:  # Encabronado
                    multiplicador = 1.8  # ¡Casi doble de fuerza!

                # Aplicar cambios a las stats del pez ANTES de pelear
                pez["Fuerza"] = int(pez["Fuerza"] * multiplicador)
                pez["Agresividad"] = int(pez["Agresividad"] * multiplicador)

                if multiplicador > 1.0:
                    info["mensaje_clima"] = (
                        f"¡Enemigo potenciado por {self.clima_actual_nombre}!"
                    )
                # ------------------------------------------------

                # 2. Procesar combate (resto del código igual)
                res = self.combate.procesar_encuentro(pez, agent_cana_rareza)
                if res["resultado"] == "victoria":
                    # Victoria: Recompensa + oro + experiencia
                    oro_ganado = int(res["recompensa"])
                    reward = 100 + oro_ganado
                    info["gold_gain"] = oro_ganado
                    info["exp_gain"] = res["experiencia"]
                    info["combat_result"] = "victoria"
                    info["enemy_name"] = pez.get("Nombre", "Enemigo")
                    # Enemigo derrotado desaparece del mapa
                    current_map[ny, nx] = GameConfig.EMPTY
                else:
                    # Derrota: Penalización pero PERMITIR movimiento (no quedar atascado)
                    info["gold_gain"] = 0
                    info["combat_result"] = "derrota"
                    info["enemy_name"] = pez.get("Nombre", "Enemigo")
                    reward = -8  # Penalización moderada
                    # CAMBIO: Limpiar enemigo para evitar loops, el agente puede pasar
                    current_map[ny, nx] = GameConfig.EMPTY
            else:
                # Ya estamos sobre un enemigo (segunda pasada), solo moverse
                current_map[ny, nx] = GameConfig.EMPTY
        elif cell == GameConfig.SHOP:
            # Vender todo el inventario
            info["venta_realizada"] = True
            reward = 5  # Recompensa por llegar a la tienda
            # Permitir movimiento libre sobre la tienda (no bloquear)

        elif cell in [GameConfig.EXIT_R, "G"]:
            req = GameConfig.get_gold_requirement(self.current_map_idx)
            if agent_gold >= req:
                reward = 200
                done = True
                info["level_complete"] = True
            else:
                # Penalización muy fuerte para evitar loops en la salida
                reward = -50
                ny, nx = y, x
                info["blocked_exit"] = True

        # IMPORTANTE: Actualizar posición
        self.agent_pos = (ny, nx)

        if self.steps >= self.max_steps:
            done = True
            info["timeout"] = True

        return self.get_vision(), reward, done, info


# ==========================================
# AGENTE (Solo inferencia)
# ==========================================


class Agent:
    def __init__(self, pkl_path, epsilon=0.0):
        self.encoder = StateEncoder()
        self.pathfinder = PathFinder()  # NUEVO: A* para navegación inteligente
        self.epsilon = epsilon  # Permitir epsilon configurable para exploración
        self.load_agent(pkl_path)
        self.gold = 0
        self.nivel = 10  # NUEVO: Nivel del agente (empieza en 10 para testing)
        self.exp = 0  # Experiencia acumulada
        self.exp_multiplicador = (
            100  # Multiplicador de experiencia (100x como en agente1.py)
        )
        self.cana_equipada = crear_cana_basica_inicial()  # Caña básica rareza 1
        self.cana_equipada.datos["rareza"] = 2  # Rareza 2 para empezar mejor
        # Inventario para objetos encontrados
        self.inventario_canas = []
        self.inventario_tesoro = None
        self.objetos_encontrados = []  # Historial de objetos
        self.max_inventario_canas = 5

        # NUEVO: Detección de loops contra paredes
        self.last_wall_actions = []  # Últimas acciones que resultaron en pared
        self.wall_attempts = 0  # Contador de intentos contra paredes

    def manejar_objeto(self, objeto):
        """Gestiona objetos encontrados: equipar, guardar o vender"""
        tipo = objeto.tipo_objeto
        oro_ganado = 0

        # Tesoro: guardar o reemplazar
        if tipo == "tesoro":
            if self.inventario_tesoro is None:
                self.inventario_tesoro = objeto
                return {"accion": "tesoro_guardado", "oro_ganado": 0}
            else:
                # Vender el de menor valor
                if objeto.precio_venta > self.inventario_tesoro.precio_venta:
                    oro_ganado = self.inventario_tesoro.precio_venta
                    self.inventario_tesoro = objeto
                    self.gold += oro_ganado
                    return {"accion": "tesoro_reemplazado", "oro_ganado": oro_ganado}
                else:
                    oro_ganado = objeto.precio_venta
                    self.gold += oro_ganado
                    return {
                        "accion": "tesoro_vendido_directo",
                        "oro_ganado": oro_ganado,
                    }

        # Caña: comparar con equipada
        if tipo in ["cana_basica", "cana_rara", "cana_epica", "cana_legendaria"]:
            # Validar nivel requerido
            if self.nivel < objeto.nivel_requerido:
                if len(self.inventario_canas) < self.max_inventario_canas:
                    self.inventario_canas.append(objeto)
                    return {"accion": "cana_guardada_nivel_bajo", "oro_ganado": 0}
                else:
                    return {"accion": "cana_descartada_nivel_bajo", "oro_ganado": 0}

            # Si no tiene caña equipada
            if self.cana_equipada is None:
                self.cana_equipada = objeto
                return {"accion": "cana_equipada", "oro_ganado": 0}

            # Comparar con caña equipada
            def es_mejor(nueva, actual):
                if nueva.rareza > actual.rareza:
                    return True
                elif nueva.rareza == actual.rareza:
                    return nueva.precio_venta > actual.precio_venta
                return False

            if es_mejor(objeto, self.cana_equipada):
                # Nueva es mejor: equipar y manejar la antigua
                cana_antigua = self.cana_equipada
                self.cana_equipada = objeto

                if len(self.inventario_canas) < self.max_inventario_canas:
                    self.inventario_canas.append(cana_antigua)
                    return {"accion": "cana_mejorada_guardada", "oro_ganado": 0}
                else:
                    # Inventario lleno: vender la peor
                    todas_canas = self.inventario_canas + [cana_antigua]
                    todas_canas.sort(key=lambda c: (c.rareza, c.precio_venta))
                    cana_a_vender = todas_canas[0]
                    self.inventario_canas = todas_canas[1:]
                    oro_ganado = cana_a_vender.precio_venta
                    self.gold += oro_ganado
                    return {
                        "accion": "cana_mejorada_vendio_peor",
                        "oro_ganado": oro_ganado,
                    }
            else:
                # Equipada es mejor: guardar o vender la nueva
                if len(self.inventario_canas) < self.max_inventario_canas:
                    self.inventario_canas.append(objeto)
                    return {"accion": "cana_guardada", "oro_ganado": 0}
                else:
                    todas_canas = self.inventario_canas + [objeto]
                    todas_canas.sort(key=lambda c: (c.rareza, c.precio_venta))
                    cana_a_vender = todas_canas[0]
                    self.inventario_canas = todas_canas[1:]
                    oro_ganado = cana_a_vender.precio_venta
                    self.gold += oro_ganado
                    return {
                        "accion": "cana_reemplazo_inventario",
                        "oro_ganado": oro_ganado,
                    }

        # Basura u otros: vender directo
        oro_ganado = objeto.precio_venta
        self.gold += oro_ganado
        return {"accion": "vendido_directo", "oro_ganado": oro_ganado}

    def vender_inventario(self):
        """Vende todo el inventario en la tienda"""
        oro_total = 0

        # Vender todas las cañas
        for cana in self.inventario_canas:
            oro_total += cana.precio_venta
        if len(self.inventario_canas) > 0:
            self.inventario_canas = []

        # Vender tesoro si hay
        if self.inventario_tesoro is not None:
            oro_total += self.inventario_tesoro.precio_venta
            self.inventario_tesoro = None

        self.gold += oro_total
        return oro_total

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

    def exp_para_siguiente_nivel(self, nivel_actual):
        """Calcula experiencia necesaria para subir de nivel (escala suave)"""
        # Fórmula: 80 * nivel^1.2
        # Nivel 2: 80 * 2^1.2 = 184 exp
        # Nivel 3: 80 * 3^1.2 = 280 exp
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

    def load_agent(self, filepath):
        print(f"\nCargando agente desde: {filepath}")

        if not os.path.exists(filepath):
            print(f"Error: No se encontró {filepath}")
            raise FileNotFoundError(f"No se encontró {filepath}")

        try:
            with open(filepath, "rb") as f:
                agent_data = pickle.load(f)

            # Cargar los Q-tables del pickle
            self.q_tables = agent_data["q_tables"]
            # NO sobrescribir epsilon (ya viene del constructor)

            total_states = sum(len(table) for table in self.q_tables)
            print(f"    {len(self.q_tables)} mapas cargados")
            print(f"    {total_states} estados aprendidos")
            print(f"    Archivo: {os.path.basename(filepath)}")

            if self.epsilon > 0:
                print(f"    Epsilon configurado: {self.epsilon} (exploración activa)")

            # VALIDACIÓN: Detectar modelo con sistema antiguo de rewards
            self._check_reward_system(agent_data)

        except Exception as e:
            print(f"Error al cargar agente: {e}")
            raise

    def _check_reward_system(self, agent_data):
        """Detecta si el modelo fue entrenado con el sistema antiguo (reward=-1 por paso)"""
        try:
            # Verificar si tiene metadata con fecha de entrenamiento
            if "timestamp" in agent_data:
                timestamp = agent_data["timestamp"]
                print(f"    Entrenado: {timestamp}")

            # Analizar Q-values promedio para detectar sistema antiguo
            all_q_values = []
            for table in self.q_tables:
                for state_q in table.values():
                    all_q_values.extend(state_q)

            if all_q_values:
                avg_q = np.mean(all_q_values)
                min_q = np.min(all_q_values)
                max_q = np.max(all_q_values)

                # Si TODOS los Q-values son negativos o promedio muy negativo
                # es probable que sea del sistema antiguo
                if max_q < 0 or (avg_q < -50 and min_q < -100):
                    print(
                        f"\n    ADVERTENCIA: Este modelo parece entrenado con el sistema ANTIGUO"
                    )
                    print(
                        f"    Q-values: avg={avg_q:.1f}, min={min_q:.1f}, max={max_q:.1f}"
                    )
                    print(f"    Sistema antiguo: reward=-1 por paso → todos negativos")
                    print(
                        f"    Sistema nuevo: reward=0 por paso → positivos hacia metas"
                    )
                    print(
                        f"\n    Solución: Re-entrenar con 'python entrenar_con_epocas.py'"
                    )
                    print(f"    Ver: GUIA_REENTRENAMIENTO.md para más detalles\n")
                else:
                    print(
                        f"    Q-values: avg={avg_q:.1f}, min={min_q:.1f}, max={max_q:.1f}"
                    )

        except Exception as e:
            print(f"    No se pudo validar sistema de rewards: {e}")

    def get_action(self, vision, map_idx, agent_pos, current_map=None):
        """
        Decide la acción usando Q-Learning + A* híbrido

        Args:
            vision: Ventana 5x5 alrededor del agente
            map_idx: Índice del mapa actual
            agent_pos: Posición actual del agente
            current_map: Mapa completo (opcional, para usar A*)
        """
        cana_rareza = self.cana_equipada.rareza if self.cana_equipada else 1
        state = self.encoder.encode(vision, self.gold, map_idx, cana_rareza, agent_pos)

        from agente1 import GameConfig

        gold_req = GameConfig.get_gold_requirement(map_idx)

        # EVITAR la salida si no tiene oro suficiente
        if self.gold < gold_req and current_map is not None:
            # Detectar si está muy cerca de la salida (distancia Manhattan < 3)
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
            # Intentar A* primero
            astar_action = self.pathfinder.get_next_action(
                current_map, agent_pos, self.gold, gold_req
            )
            if astar_action is not None:
                return astar_action

            # Si A* falla, usar dirección directa hacia salida
            exit_positions = np.where((current_map == "R") | (current_map == "G"))
            if len(exit_positions[0]) > 0:
                exit_y, exit_x = exit_positions[0][0], exit_positions[1][0]
                agent_y, agent_x = agent_pos
                dy = exit_y - agent_y
                dx = exit_x - agent_x

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
                    return moves_priority[0]

            # Último recurso: cualquier dirección válida
            surroundings = state[1]
            valid = [i for i in range(4) if surroundings[i] != 1]
            if valid:
                return np.random.choice(valid)

            # TERMINAR AQUÍ - No continuar con otras lógicas si tiene oro suficiente
            return np.random.randint(4)

        # A partir de aquí, solo ejecutar si NO tiene oro suficiente

        if map_idx >= len(self.q_tables):
            return np.random.randint(4)  # Fallback

        table = self.q_tables[map_idx]

        # CRÍTICO: Detectar si está atascado contra paredes (solo si necesita oro)
        if len(self.last_wall_actions) >= 2:  # Reducido de 3 a 2 para detectar antes
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

        # SIEMPRE buscar objetivos cuando necesita oro (antes de consultar Q-table)
        if self.gold < gold_req and current_map is not None:
            objetivo_pos = self._buscar_objetivo_cercano(current_map, agent_pos)
            if objetivo_pos is not None:
                # Calcular dirección hacia el objetivo
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

        # Si no encuentra objetivos o estado desconocido, continuar con lógica normal
        if state not in table:
            # Estado nunca visto - buscar objetivos activamente
            if current_map is not None:
                # BUSCAR OBJETIVOS cuando no tiene oro suficiente
                if self.gold < gold_req:
                    # Prioridad: Tesoros > Tiendas (si tiene inventario) > Enemigos
                    objetivo_pos = self._buscar_objetivo_cercano(current_map, agent_pos)
                    if objetivo_pos is not None:
                        # Calcular dirección hacia el objetivo
                        obj_y, obj_x = objetivo_pos
                        agent_y, agent_x = agent_pos
                        dy = obj_y - agent_y
                        dx = obj_x - agent_x

                        surroundings = state[1]  # (N, S, E, O)
                        moves_priority = []

                        # Priorizar movimiento según distancia mayor
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
                            return moves_priority[0]

                # Si no hay objetivos cercanos, usar A*
                astar_action = self.pathfinder.get_next_action(
                    current_map, agent_pos, self.gold, gold_req
                )
                if astar_action is not None:
                    return astar_action

            # Si A* no funciona, usar heurística mejorada
            surroundings = state[1]  # (N, S, E, O)
            on_exit = state[3]

            # Si estamos en la salida y tenemos oro, intentar quedarse
            if on_exit and state[0]:  # can_exit
                return np.random.randint(4)

            # Priorizar direcciones con tesoros > enemigos > espacios vacíos
            moves_tesoro = []
            moves_enemigo = []
            moves_vacio = []

            for i, val in enumerate(surroundings):
                if val == 2:  # Tesoro
                    moves_tesoro.append(i)
                elif val == 4:  # Enemigo (X o E)
                    moves_enemigo.append(i)
                elif val == 0:  # Espacio vacío
                    moves_vacio.append(i)

            # Prioridad: tesoros > enemigos > vacíos
            if moves_tesoro:
                return np.random.choice(moves_tesoro)
            if moves_enemigo and self.gold < gold_req:  # Solo si necesita oro
                return np.random.choice(moves_enemigo)
            if moves_vacio:
                return np.random.choice(moves_vacio)

            # Si todo bloqueado, elegir cualquier dirección no-pared
            moves = [i for i, val in enumerate(surroundings) if val != 1]
            if moves:
                return np.random.choice(moves)

            return np.random.randint(4)

        # Política epsilon-greedy (permite exploración)
        if np.random.rand() < self.epsilon:
            # 50% usa A*, 50% aleatorio durante exploración
            if current_map is not None and np.random.rand() < 0.5:
                from agente1 import GameConfig

                gold_req = GameConfig.get_gold_requirement(map_idx)
                astar_action = self.pathfinder.get_next_action(
                    current_map, agent_pos, self.gold, gold_req
                )
                if astar_action is not None:
                    return astar_action

            # Exploración: elegir acción aleatoria (preferir no-paredes)
            surroundings = state[1]
            moves = [i for i, val in enumerate(surroundings) if val != 1]
            if moves:
                return np.random.choice(moves)
            return np.random.randint(4)

        # Explotación: elegir mejor acción conocida
        # NUEVA PRIORIDAD: Si necesita oro, buscar objetivos INCLUSO si estado es conocido
        if self.gold < gold_req and current_map is not None:
            objetivo_pos = self._buscar_objetivo_cercano(
                current_map, agent_pos, radio=12
            )
            if objetivo_pos is not None:
                # Calcular dirección hacia el objetivo
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
                    # SIEMPRE usar búsqueda cuando hay objetivo visible
                    return moves_priority[0]

        # Si no hay objetivos visibles, usar Q-table
        # Filtrar paredes ANTES de elegir según Q-values
        surroundings = state[1]  # (N, S, E, O)
        valid_actions = [i for i in range(4) if surroundings[i] != 1]  # No-paredes

        if not valid_actions:
            # Todas son paredes (no debería pasar, pero por seguridad)
            return np.random.randint(4)

        # Filtrar Q-values solo de acciones válidas (no-paredes)
        valid_q_values = [(i, table[state][i]) for i in valid_actions]
        max_val = max(v for _, v in valid_q_values)

        # Si Q-values son muy bajos O necesita oro, usar A* como guía
        if (max_val <= 0.1 or self.gold < gold_req) and current_map is not None:
            from agente1 import GameConfig

            gold_req = GameConfig.get_gold_requirement(map_idx)
            astar_action = self.pathfinder.get_next_action(
                current_map, agent_pos, self.gold, gold_req
            )
            if astar_action is not None:
                return astar_action

        # Elegir entre las mejores acciones válidas (no-paredes)
        best_actions = [i for i, v in valid_q_values if v == max_val]
        return np.random.choice(best_actions)


# VISUALIZADOR MEJORADO


def render_game(env, agent, stats, debug_info=None):
    """Renderiza el estado del juego con interfaz mejorada"""
    os.system("cls" if os.name == "nt" else "clear")

    # Header
    print(f"{' VISUALIZADOR DE AGENTE ENTRENADO':^70}")

    # Stats superior
    req = GameConfig.get_gold_requirement(env.current_map_idx)
    print(f"\n MAPA: {env.current_map_idx + 1}/{len(env.maps)}")
    print(f" ORO: {agent.gold:4d} / {req:4d} {'✓' if agent.gold >= req else '✗'}")

    # Mostrar información de caña y nivel
    cana_rareza = agent.cana_equipada.rareza if agent.cana_equipada else 1
    cana_nombre = agent.cana_equipada.tipo_objeto if agent.cana_equipada else "ninguna"

    # Calcular probabilidad de victoria contra enemigos
    prob_enemigo_debil = min(100, int((cana_rareza * 12) / 20 * 100))  # vs Fuerza 20
    prob_enemigo_medio = min(100, int((cana_rareza * 12) / 35 * 100))  # vs Fuerza 35

    # Barra de experiencia
    exp_actual = agent.exp
    exp_necesaria = agent.exp_para_siguiente_nivel(agent.nivel)
    porcentaje_exp = (exp_actual / exp_necesaria * 100) if exp_necesaria > 0 else 0
    barras = int(porcentaje_exp / 10)  # 10 bloques máximo
    barra_exp = "█" * barras + "░" * (10 - barras)

    print(f" NIVEL: {agent.nivel}/100")
    print(f" EXP: [{barra_exp}] {exp_actual}/{exp_necesaria} ({porcentaje_exp:.0f}%)")
    print(
        f" CAÑA: {cana_nombre} (Rareza {cana_rareza}) | Bonus: +{cana_rareza * 5} oro"
    )
    print(f"       VS débil: ~{prob_enemigo_debil}% | VS medio: ~{prob_enemigo_medio}%")
    print(f" PASOS: {env.steps}/{env.max_steps}")

    # Mostrar clima, día/noche y efectos
    periodo = "🌅 DÍA" if env.es_dia else "🌙 NOCHE"
    clima_emoji = {"Soleado": "☀️", "Lluvia": "🌧️", "Tormenta": "⛈️", "Templado": "🌤️"}
    emoji = clima_emoji.get(env.clima_actual_nombre, "🌤️")

    if env.estado_furia == 1:
        efecto = "Enemigos normales"
    elif env.estado_furia == 3:
        efecto = "⚡ Enemigos enojados (+30% fuerza)"
    else:  # estado_furia == 6
        efecto = "⚠️ ENEMIGOS ENCABRONADOS (+80% fuerza)"

    print(f" TIEMPO: {periodo} | {emoji} {env.clima_actual_nombre}")
    print(f" EFECTO: {efecto}")

    # Mostrar inventario de objetos encontrados
    if len(agent.objetos_encontrados) > 0:
        print(f" OBJETOS ENCONTRADOS: {len(agent.objetos_encontrados)}")
        # Mostrar últimos 3 objetos
        ultimos = agent.objetos_encontrados[-3:]
        print(f"   Últimos: {', '.join(ultimos)}")

    # Mostrar inventario actual
    if len(agent.inventario_canas) > 0 or agent.inventario_tesoro:
        print(f" INVENTARIO:")
        if len(agent.inventario_canas) > 0:
            print(
                f"   Cañas: {len(agent.inventario_canas)}/{agent.max_inventario_canas} (Valor: {sum(c.precio_venta for c in agent.inventario_canas)} oro)"
            )
        if agent.inventario_tesoro:
            print(
                f"   Tesoro: {agent.inventario_tesoro.tipo_objeto} (Valor: {agent.inventario_tesoro.precio_venta} oro)"
            )

    # Mostrar estadísticas de combate
    if stats.get("combats_won", 0) > 0 or stats.get("combats_lost", 0) > 0:
        total_combats = stats["combats_won"] + stats["combats_lost"]
        win_rate = (
            (stats["combats_won"] / total_combats * 100) if total_combats > 0 else 0
        )
        print(
            f" COMBATES: {stats['combats_won']}V/{stats['combats_lost']}D (Tasa: {win_rate:.0f}%)"
        )

    # Mejorar display de reward
    reward_total = stats["total_reward"]
    avg_reward = reward_total / env.steps if env.steps > 0 else 0
    print(f" REWARD TOTAL: {reward_total:.0f} (Promedio: {avg_reward:.2f}/paso)")

    # Indicador de eficiencia
    if env.steps > 20:
        if avg_reward >= 0.3:  # Cerca del +0.5 base
            print(f"    Navegación excelente (pocas colisiones)")
        elif avg_reward >= -1.0:
            print(f"    Navegación aceptable (algunas colisiones, -3 c/u)")
        else:
            collisions_estimated = abs(int(reward_total / 3))
            print(
                f"    Muchas colisiones (~{collisions_estimated} choques con paredes)"
            )

    print(f"  MAPAS COMPLETADOS: {stats['maps_cleared']}")

    # DEBUG INFO
    # if debug_info:
    #     print(
    #         f"\nDEBUG: Conocidos={debug_info['known']}, Desconocidos={debug_info['unknown']}, Pos={env.agent_pos}"
    #     )

    print("\n" + "-" * 70)

    m = env.map_templates[env.current_map_idx].copy()

    # Superponer cambios del mapa actual (tesoros consumidos)
    current_map = env.maps[env.current_map_idx]
    for i in range(18):
        for j in range(18):
            # Si en el mapa actual hay espacio/comilla pero en template había T, se consumió
            if current_map[i, j] in [" ", "'"] and m[i, j] == "T":
                m[i, j] = current_map[i, j]  # Mostrar como consumido

    # Mostrar agente
    ay, ax = env.agent_pos
    original_cell = m[ay, ax]

    if original_cell == "R":
        m[ay, ax] = "◉"  # Círculo para indicar que está sobre la salida
    else:
        m[ay, ax] = "A"

    # Leyenda lateral
    legend = [
        "LEYENDA:",
        "A = Agente",
        "◉ = Agente en salida",
        "S = Inicio",
        "R/G = Salida",
        "T = Tesoro",
        "P = Tienda",
        "X/E = Enemigo",
    ]

    print("    " + "".join([f"{i:2d}" for i in range(18)]))  # FIX: 18 columnas
    for i, row in enumerate(m):
        line = f"{i:2d}  " + "".join(row)
        if i < len(legend):
            line += f"    {legend[i]}"
        print(line)

    print("-" * 70)


def main():
    print("\n" + "=" * 70)
    print(f"{'VISUALIZADOR DE AGENTE Q-LEARNING':^70}")
    print("=" * 70 + "\n")

    print("CONFIGURACIÓN MEJORADA:")
    print("   • Nivel de caña inicial: 2 (antes: 1)")
    print("   • Mejor oportunidad contra enemigos (~60% vs débiles, ~35% vs medios)")
    print("   • Bonus de tesoros: +20 oro (antes: +15 oro)")
    print()

    # CONFIGURACIÓN

    # Solicitar rutas
    print("Tipos de mapas:")
    print("   • 'mapa' - Mapas complejos pre-diseñados (laberintos)")
    print("   • 'mapa_aleatorio_prueba' - Mapas aleatorios similares al entrenamiento")
    maps_folder = (
        input("\nCarpeta de mapas (default: 'mapa_aleatorio_prueba'): ").strip()
        or "mapa_aleatorio_prueba"
    )

    # Generar mapas si no existen
    if not generar_mapas_si_no_existen(maps_folder):
        print("\nNo se pudieron cargar o generar mapas")
        sys.exit(1)

    pkl_path = (
        input(" Archivo del agente (default: 'agente_entrenado_epocas.pkl'): ").strip()
        or "agente_entrenado_epocas.pkl"
    )

    # Velocidad de reproducción
    print("\nVelocidad de reproducción:")
    print("   1 = Muy rápida (0.05s)")
    print("   2 = Rápida (0.1s)")
    print("   3 = Normal (0.2s)")
    print("   4 = Lenta (0.5s)")
    speed_choice = input("Selecciona (1-4, default: 2): ").strip() or "2"

    speeds = {"1": 0.05, "2": 0.1, "3": 0.2, "4": 0.5}
    delay = speeds.get(speed_choice, 0.1)

    # Modo de exploración
    print("\nModo de exploración:")
    print("   0 = Sin exploración (epsilon=0, solo greedy)")
    print("   1 = Exploración baja (epsilon=0.05)")
    print("   2 = Exploración media (epsilon=0.15)")
    print("   3 = Exploración alta (epsilon=0.30)")
    epsilon_choice = input("Selecciona (0-3, default: 0): ").strip() or "0"

    epsilon_values = {"0": 0.0, "1": 0.05, "2": 0.15, "3": 0.30}
    epsilon = epsilon_values.get(epsilon_choice, 0.0)

    if epsilon > 0:
        print(f"   Modo exploración activado (epsilon={epsilon})")
        print(f"   El agente tomará acciones aleatorias {epsilon*100:.0f}% del tiempo")

    # INICIALIZACIÓN

    try:
        env = SimpleMapEnvironment(maps_folder=maps_folder)
        agent = Agent(pkl_path=pkl_path, epsilon=epsilon)  # 🔧 Pasar epsilon
    except Exception as e:
        print(f"\n Error al cargar: {e}")
        sys.exit(1)

    print(f"\n Todo listo. Iniciando visualización...\n")
    time.sleep(2)

    # LOOP DE VISUALIZACIÓN

    vision = env.reset(keep_progress=False)
    agent.gold = 0
    # Ya no necesitamos resetear rod_lvl, el agente tiene su nivel y caña

    stats = {
        "total_reward": 0,
        "maps_cleared": 0,
        "total_steps": 0,
        "treasures_collected": 0,
        "objects_rejected": 0,
        "upgrades_bought": 0,
        "combats_won": 0,
        "combats_lost": 0,
    }

    max_total_steps = 10000

    try:
        known_states = 0
        unknown_states = 0
        position_history = []  # Rastrear posiciones para detectar bucles

        while stats["total_steps"] < max_total_steps:
            # Verificar estado
            cana_rareza = agent.cana_equipada.rareza if agent.cana_equipada else 1
            state = agent.encoder.encode(
                vision, agent.gold, env.current_map_idx, cana_rareza, env.agent_pos
            )
            is_known = state in agent.q_tables[env.current_map_idx]
            if is_known:
                known_states += 1
            else:
                unknown_states += 1

            # DETECCIÓN AGRESIVA DE INMOBILIDAD Y BUCLES
            position_history.append(env.agent_pos)
            if len(position_history) > 10:
                position_history.pop(0)

            # Detectar inmobilidad y ciclos
            is_stuck_in_loop = False
            if len(position_history) >= 4:
                # Inmóvil: misma posición 4 veces seguidas
                if all(pos == position_history[-1] for pos in position_history[-4:]):
                    is_stuck_in_loop = True
                # Ciclo de 2 posiciones (A->B->A->B)
                elif len(position_history) >= 4 and (
                    position_history[-1] == position_history[-3]
                    and position_history[-2] == position_history[-4]
                    and position_history[-1] != position_history[-2]
                ):
                    is_stuck_in_loop = True

            # Si detectamos bucle, forzar escape inmediato
            if is_stuck_in_loop:
                # Eliminar estados problemáticos de la Q-table
                cana_rareza = agent.cana_equipada.rareza if agent.cana_equipada else 1
                for pos in set(position_history[-4:]):
                    temp_vision = env.get_vision()
                    state = agent.encoder.encode(
                        temp_vision, agent.gold, env.current_map_idx, cana_rareza, pos
                    )
                    if state in agent.q_tables[env.current_map_idx]:
                        if not hasattr(agent, "_blocked_states"):
                            agent._blocked_states = {}
                        agent._blocked_states[state] = agent.q_tables[
                            env.current_map_idx
                        ][state].copy()
                        del agent.q_tables[env.current_map_idx][state]

                position_history = []  # Reset completo
                position_history = []  # Reset completo
                time.sleep(1)

            # Renderizar con debug info
            debug_info = {"known": known_states, "unknown": unknown_states}
            render_game(env, agent, stats, debug_info)

            # Decisión del agente (pasar mapa completo para A*)
            action = agent.get_action(
                vision,
                env.current_map_idx,
                env.agent_pos,
                env.maps[env.current_map_idx],
            )

            # Ejecutar acción
            cana_rareza = agent.cana_equipada.rareza if agent.cana_equipada else 1
            vision, reward, done, info = env.step(
                action, agent.gold, agent.nivel, cana_rareza
            )

            # NUEVO: Tracking de acciones contra paredes
            if reward == -3:  # Chocó contra pared
                agent.last_wall_actions.append(action)
                agent.wall_attempts += 1
                if len(agent.last_wall_actions) > 5:
                    agent.last_wall_actions.pop(0)  # Mantener solo últimas 5
            else:
                # Reset si logró moverse exitosamente
                if reward >= 0:
                    agent.last_wall_actions = []
                    agent.wall_attempts = 0

            # Mostrar mensaje de combate si ocurrió uno
            if "combat_result" in info:
                enemy_name = info.get("enemy_name", "Enemigo")
                if info["combat_result"] == "victoria":
                    oro_ganado = info.get("gold_gain", 0)
                    exp_ganada = info.get("exp_gain", 0)
                    print(
                        f"\n¡VICTORIA contra {enemy_name}! +{oro_ganado} oro, +{exp_ganada} exp"
                    )
                    # Procesar experiencia ganada
                    if exp_ganada > 0:
                        niveles_subidos, exp_real = agent.agregar_experiencia(
                            exp_ganada
                        )
                        if niveles_subidos > 0:
                            print(
                                f"¡NIVEL {agent.nivel}! (+{niveles_subidos} niveles, +{exp_real} exp total)"
                            )
                else:
                    print(f"\nDERROTA contra {enemy_name}... El enemigo sigue ahí")
                time.sleep(1.5)  # Pausa extra para ver resultado de combate
            # Mostrar mensaje de objeto encontrado
            if "objeto_encontrado" in info:
                objeto_nombre = info.get("objeto_nombre", "Objeto")
                if info.get("objeto_rechazado") == "nivel_insuficiente":
                    nivel_req = info.get("nivel_requerido", "?")
                    stats["objects_rejected"] += 1
                    print(
                        f"\n{objeto_nombre} encontrado pero nivel insuficiente (necesitas nivel {nivel_req})"
                    )
                    time.sleep(1.5)
                elif info.get("objeto_recogido"):
                    rareza = info.get("objeto_rareza", 1)
                    valor = info.get("objeto_valor", 0)
                    objeto_tipo = info.get("objeto_tipo", "objeto")

                    # Manejar el objeto usando el sistema completo
                    resultado = agent.manejar_objeto(info["objeto_encontrado"])
                    accion = resultado["accion"]
                    oro_extra = resultado["oro_ganado"]

                    # Guardar en historial
                    agent.objetos_encontrados.append(objeto_nombre)

                    # Mensajes según la acción
                    if accion == "cana_mejorada_guardada":
                        print(
                            f"\n¡{objeto_nombre} EQUIPADO! (Rareza {rareza}) - Caña antigua guardada"
                        )
                    elif accion == "cana_mejorada_vendio_peor":
                        print(
                            f"\n¡{objeto_nombre} EQUIPADO! (Rareza {rareza}) - Vendiste la peor por {oro_extra} oro"
                        )
                    elif accion == "cana_equipada":
                        print(f"\n¡{objeto_nombre} EQUIPADO! (Rareza {rareza})")
                    elif accion == "cana_guardada":
                        print(
                            f"\n{objeto_nombre} guardado en inventario ({len(agent.inventario_canas)}/{agent.max_inventario_canas})"
                        )
                    elif accion == "cana_reemplazo_inventario":
                        print(
                            f"\n{objeto_nombre} guardado - Vendiste la peor por {oro_extra} oro"
                        )
                    elif accion == "tesoro_guardado":
                        print(f"\n{objeto_nombre} guardado (Valor: {valor} oro)")
                    elif accion == "tesoro_reemplazado":
                        print(
                            f"\n{objeto_nombre} guardado - Vendiste tesoro anterior por {oro_extra} oro"
                        )
                    elif accion in ["tesoro_vendido_directo", "vendido_directo"]:
                        print(f"\n{objeto_nombre} vendido por {oro_extra} oro")
                    else:
                        print(
                            f"\n¡{objeto_nombre} recogido! (Rareza {rareza}, Valor: {valor} oro)"
                        )

                    time.sleep(1.5)
            # Actualizar stats del agente
            if "gold_gain" in info:
                agent.gold += info["gold_gain"]
                if info.get("combat_result"):
                    # Es combate, no tesoro
                    if info["combat_result"] == "victoria":
                        stats["combats_won"] += 1
                    else:
                        stats["combats_lost"] += 1
                # Si no hay combat_result, no incrementar treasures_collected
                # porque los tesoros ahora generan objetos, no oro directo

            if info.get("objeto_recogido"):
                stats["treasures_collected"] += 1

            # Manejar venta en tienda
            if info.get("venta_realizada"):
                oro_ganado = agent.vender_inventario()
                if oro_ganado > 0:
                    print(f"\nVendiste todo tu inventario por {oro_ganado} oro")
                    print(f"Cañas vendidas: {len(agent.inventario_canas)}")
                    if agent.inventario_tesoro:
                        print(f"Tesoro vendido")
                    print(f"Oro total: {agent.gold}")
                    time.sleep(1.5)
                else:
                    print(f"\nVisitaste la tienda pero no tenías nada que vender")
                    time.sleep(1)

            stats["total_reward"] += reward
            stats["total_steps"] += 1

            time.sleep(delay)

            # Manejar fin de nivel/juego
            if done:
                render_game(env, agent, stats, debug_info)

                if info.get("level_complete"):
                    print(f"{'¡NIVEL COMPLETADO!':^70}")

                    stats["maps_cleared"] += 1
                    time.sleep(2)

                    env.current_map_idx += 1
                    if env.current_map_idx >= len(env.maps):
                        print("\n" + "" * 35)
                        print(f"{'¡¡¡ VICTORIA TOTAL !!!':^70}")
                        print("" * 35)
                        print(f"\n ESTADÍSTICAS FINALES:")
                        print(f"   • Mapas completados: {stats['maps_cleared']}")
                        print(
                            f"   • Objetos recolectados: {stats['treasures_collected']}"
                        )
                        print(
                            f"   • Objetos rechazados (nivel bajo): {stats['objects_rejected']}"
                        )
                        print(f"   • Combates ganados: {stats['combats_won']}")
                        print(f"   • Combates perdidos: {stats['combats_lost']}")
                        if stats["combats_won"] + stats["combats_lost"] > 0:
                            win_rate = (
                                stats["combats_won"]
                                / (stats["combats_won"] + stats["combats_lost"])
                                * 100
                            )
                            print(f"   • Tasa de victoria en combate: {win_rate:.1f}%")
                        print(f"   • Mejoras compradas: {stats['upgrades_bought']}")
                        print(f"   • Pasos totales: {stats['total_steps']}")
                        print(f"   • Reward total: {stats['total_reward']:.0f}")
                        break

                    vision = env.reset(keep_progress=True)

                elif info.get("timeout"):
                    print("\n" + " " * 35)
                    print(f"{'TIMEOUT - Se acabó el tiempo':^70}")
                    print(" " * 35)
                    time.sleep(2)
                    break

                else:
                    print("\n" + "" * 35)
                    print(f"{'GAME OVER':^70}")
                    print("" * 35)
                    print(f"\n ESTADÍSTICAS:")
                    print(f"   • Mapas completados: {stats['maps_cleared']}")
                    print(f"   • Objetos recolectados: {stats['treasures_collected']}")
                    print(f"   • Objetos rechazados: {stats['objects_rejected']}")
                    print(f"   • Combates ganados: {stats['combats_won']}")
                    print(f"   • Combates perdidos: {stats['combats_lost']}")
                    print(f"   • Pasos totales: {stats['total_steps']}")
                    print(f"   • Reward total: {stats['total_reward']:.0f}")
                    break

    except KeyboardInterrupt:
        print("\n\n Visualización interrumpida por usuario")
        print(f"\n ESTADÍSTICAS PARCIALES:")
        print(f"   • Mapas completados: {stats['maps_cleared']}")
        print(f"   • Pasos totales: {stats['total_steps']}")
        print(f"   • Objetos recolectados: {stats['treasures_collected']}")
        print(f"   • Objetos rechazados: {stats['objects_rejected']}")
        print(f"   • Combates ganados: {stats['combats_won']}")
        print(f"   • Combates perdidos: {stats['combats_lost']}")
        if stats["combats_won"] + stats["combats_lost"] > 0:
            win_rate = (
                stats["combats_won"]
                / (stats["combats_won"] + stats["combats_lost"])
                * 100
            )
            print(f"   • Tasa de victoria: {win_rate:.1f}%")

    print(f"{' Visualización finalizada':^70}")


# PUNTO DE ENTRADA

if __name__ == "__main__":
    main()
