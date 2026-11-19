"""
Script de entrenamiento con escenarios aleatorios por época
Cada época genera 12 nuevos escenarios resolubles para entrenar al agente
"""

import numpy as np
import os
from datetime import datetime

from agente1 import (
    MapEnvironment,
    PersistentAgent,
    GameConfig,
    crear_cana_basica_inicial,
)
from generador_aleatorio import GeneradorEscenariosAleatorios


def entrenar_por_epocas(num_epocas=30, episodios_por_epoca=10):
    """
    Entrena al agente con escenarios aleatorios generados por época

    CONFIGURACIÓN ACTUAL: ENTRENAMIENTO BÁSICO
    - 30 épocas x 10 episodios = 300 episodios totales
    - Tiempo: ~5-10 minutos
    - Para entrenamiento PESADO (mejor PC): cambiar a (100, 30) = 3000 episodios

    Args:
        num_epocas: Número de épocas de entrenamiento
        episodios_por_epoca: Episodios que el agente juega en cada época
    """
    print("=" * 70)
    print("  ENTRENAMIENTO CON ESCENARIOS ALEATORIOS POR ÉPOCA")
    print("=" * 70)
    print(f"  Épocas: {num_epocas}")
    print(f"  Episodios por época: {episodios_por_epoca}")
    print(f"  Total de episodios: {num_epocas * episodios_por_epoca}")
    print("=" * 70 + "\n")

    # Crear generador de escenarios
    generador = GeneradorEscenariosAleatorios(seed=42)

    # Crear carpeta para guardar escenarios
    carpeta_epocas = "mapas_entrenamiento"
    if not os.path.exists(carpeta_epocas):
        os.makedirs(carpeta_epocas)

    # Inicializar agente (persiste entre épocas)
    agent = PersistentAgent(num_maps=12)

    # Estadísticas globales
    estadisticas_epocas = {
        "rewards_por_epoca": [],
        "mapas_completados_por_epoca": [],
        "oro_acumulado_por_epoca": [],
        "nivel_agente_por_epoca": [],
    }

    episodio_global = 0

    try:
        for epoca in range(num_epocas):
            print(f"\nÉPOCA {epoca + 1}/{num_epocas}", end=" ")

            # 1. GENERAR 12 ESCENARIOS ALEATORIOS PARA ESTA ÉPOCA
            escenarios = generador.generar_12_escenarios_epoca(epoca)

            # Verificar que sean resolubles (sin print detallado)
            resolubles, no_resolubles = generador.verificar_resolubles(escenarios)
            print(f"| Escenarios: {len(resolubles)}/12", end="")

            # 2. GUARDAR ESCENARIOS DE LA ÉPOCA
            generador.guardar_escenarios_epoca(epoca, escenarios, carpeta_epocas)

            # 3. CREAR ENTORNO CON LOS NUEVOS ESCENARIOS (modo silencioso)
            carpeta_epoca_actual = os.path.join(carpeta_epocas, f"epoca_{epoca:04d}")
            env = MapEnvironment(
                maps_folder=carpeta_epoca_actual, use_random=False, verbose=False
            )

            # RESETEAR AGENTE PARA NUEVA ÉPOCA (mantener Q-tables, resetear recursos)
            agent.gold = 0
            agent.nivel = 1
            agent.exp = 0
            agent.cana_equipada = crear_cana_basica_inicial()
            agent.inventario_canas = []
            agent.inventario_tesoro = None
            agent.last_positions = []

            # Estadísticas de esta época
            rewards_epoca = []
            mapas_completados_epoca = []
            oro_epoca = []

            # 4. ENTRENAR DURANTE episodios_por_epoca
            for episodio_epoca in range(episodios_por_epoca):
                episodio_global += 1

                # Reset del entorno Y del agente para cada episodio
                vision = env.reset(keep_progress=False)

                # RESETEAR RECURSOS DEL AGENTE (mantener Q-tables)
                agent.gold = 0
                agent.nivel = 1
                agent.exp = 0
                agent.cana_equipada = crear_cana_basica_inicial()
                agent.inventario_canas = []
                agent.inventario_tesoro = None
                agent.last_positions = []

                total_reward = 0
                maps_cleared = 0

                # Límite de pasos por episodio (reducido de 5000 a 2000)
                max_steps_episodio = 2000
                steps_episodio = 0

                # Timeout por mapa individual (300 pasos máximo)
                steps_mapa_actual = 0
                max_steps_por_mapa = 300
                mapa_actual = env.current_map_idx

                while steps_episodio < max_steps_episodio:
                    current_map = env.current_map_idx

                    # Detectar cambio de mapa para resetear contador
                    if current_map != mapa_actual:
                        mapa_actual = current_map
                        steps_mapa_actual = 0

                    # Timeout por mapa: si pasa mucho tiempo sin completar, abortar episodio
                    steps_mapa_actual += 1
                    if steps_mapa_actual > max_steps_por_mapa:
                        done = True
                        break

                    old_pos = env.agent_pos

                    # Decidir acción (pasar mapa completo para A*)
                    action = agent.get_action(
                        vision,
                        current_map,
                        env.agent_pos,
                        env.maps[env.current_map_idx],
                    )

                    # Actuar (pasar nivel y rareza de caña)
                    cana_rareza = (
                        agent.cana_equipada.rareza if agent.cana_equipada else 1
                    )
                    next_vision, reward, done, info = env.step(
                        action, agent.gold, agent.nivel, cana_rareza
                    )

                    # Actualizar oro
                    if "gold_gain" in info:
                        agent.gold += info["gold_gain"]
                        if agent.gold < 0:
                            agent.gold = 0

                    # Ganar experiencia
                    if "exp_gain" in info:
                        niveles_subidos, _ = agent.agregar_experiencia(info["exp_gain"])
                        if niveles_subidos > 0 and episodio_epoca % 10 == 0:
                            print(f"Nivel {agent.nivel} alcanzado")

                    # Aprender (Q-learning)
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

                    # Gestionar objetos
                    if "objeto_encontrado" in info and info.get("objeto_recogido"):
                        resultado = agent.gestionar_objeto(info["objeto_encontrado"])
                        if resultado["oro_ganado"] > 0:
                            agent.gold += resultado["oro_ganado"]
                        agent.training_stats["objetos_recogidos"] += 1

                    # Vender en tienda
                    if info.get("venta_realizada"):
                        oro_vendido = agent.vender_inventario()
                        agent.gold += oro_vendido

                    vision = next_vision
                    total_reward += reward
                    steps_episodio += 1

                    if done:
                        # Verificar si completó el nivel
                        if info.get("level_complete"):
                            maps_cleared += 1
                            mapa_completado = env.current_map_idx

                            # Verificar victoria total
                            if mapa_completado == 11 and agent.tiene_mejor_cana():
                                print(f"\n¡VICTORIA EN EPISODIO {episodio_epoca}!")
                                # Fin de este episodio victorioso
                                break
                            else:
                                # Avanzar al siguiente mapa
                                env.current_map_idx += 1

                                if env.current_map_idx >= len(env.maps):
                                    # Completó todos los mapas, terminar episodio
                                    break
                                else:
                                    # Continuar al siguiente mapa
                                    vision = env.reset(keep_progress=True)
                                    agent.last_positions = []

                            done = False
                        else:
                            # Timeout o game over
                            break

                # Guardar estadísticas del episodio
                rewards_epoca.append(total_reward)
                mapas_completados_epoca.append(maps_cleared)
                oro_epoca.append(agent.gold)

                # Actualizar epsilon
                agent.epsilon = max(
                    agent.epsilon * agent.epsilon_decay, agent.epsilon_min
                )

                # Log reducido: solo cada 20 episodios
                if episodio_epoca % 20 == 0 and episodio_epoca > 0:
                    avg_reward = (
                        np.mean(rewards_epoca[-20:])
                        if len(rewards_epoca) >= 20
                        else total_reward
                    )
                    avg_mapas = (
                        np.mean(mapas_completados_epoca[-20:])
                        if len(mapas_completados_epoca) >= 20
                        else maps_cleared
                    )
                    # Solo imprimir en la misma línea, sin nueva línea
                    print(".", end="", flush=True)

            # Estadísticas de la época
            estadisticas_epocas["rewards_por_epoca"].append(np.mean(rewards_epoca))
            estadisticas_epocas["mapas_completados_por_epoca"].append(
                np.mean(mapas_completados_epoca)
            )
            estadisticas_epocas["oro_acumulado_por_epoca"].append(np.mean(oro_epoca))
            estadisticas_epocas["nivel_agente_por_epoca"].append(agent.nivel)

            # Resumen compacto en una línea
            print(
                f" | Reward: {np.mean(rewards_epoca):.0f} | Mapas: {np.mean(mapas_completados_epoca):.1f} | "
                f"Nivel: {agent.nivel} | ε: {agent.epsilon:.3f}"
            )

            # Guardar agente cada 10 épocas (sin print)
            if (epoca + 1) % 10 == 0:
                filename = f"agente_epoca_{epoca+1}.pkl"
                agent.save_agent(filename)
                print(f" [✓ Guardado: {filename}]")

    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por usuario")

    # GUARDAR RESULTADOS FINALES
    print("\n" + "=" * 70)
    print("   GUARDANDO RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 70 + "\n")

    # Guardar agente final
    agent.save_agent("agente_entrenado_epocas.pkl")
    agent.export_qtable_json("qtables_epocas.json")

    # Resumen final
    print("\n" + "=" * 70)
    print("   RESUMEN FINAL DEL ENTRENAMIENTO")
    print("=" * 70)
    print(f"  Épocas completadas: {len(estadisticas_epocas['rewards_por_epoca'])}")
    print(f"  Episodios totales: {episodio_global}")
    print(f"  Nivel final del agente: {agent.nivel}/100")
    if agent.cana_equipada:
        print(
            f"  Caña equipada: {agent.cana_equipada.tipo_objeto} (Rareza {agent.cana_equipada.rareza})"
        )
    else:
        print(f"  Caña equipada: ninguna")
    print(f"  Estados aprendidos: {sum(len(table) for table in agent.q_tables)}")
    print(f"  Epsilon final: {agent.epsilon:.4f}")
    print("=" * 70 + "\n")

    return agent, estadisticas_epocas


if __name__ == "__main__":
    # CONFIGURACIÓN OPTIMIZADA PARA 5 HORAS
    # Prioriza VARIEDAD de mapas (50 épocas = 600 mapas únicos)
    # sobre repetición (100 episodios por época es suficiente)
    NUM_EPOCAS = 50  # Máxima variedad: 50 × 12 = 600 mapas diferentes
    EPISODIOS_POR_EPOCA = 100  # Suficiente para aprender cada conjunto

    # ALTERNATIVAS según tu tiempo disponible:
    # 3 horas:  NUM_EPOCAS = 30,  EPISODIOS_POR_EPOCA = 100
    # 5 horas:  NUM_EPOCAS = 50,  EPISODIOS_POR_EPOCA = 100  ← ACTUAL
    # 8 horas:  NUM_EPOCAS = 80,  EPISODIOS_POR_EPOCA = 100
    # 10 horas: NUM_EPOCAS = 100, EPISODIOS_POR_EPOCA = 100

    print("\n" + "=" * 70)
    print("  CONFIGURACIÓN DE ENTRENAMIENTO - OPTIMIZADO 5 HORAS")
    print("=" * 70)
    print(
        f"{NUM_EPOCAS} épocas x {EPISODIOS_POR_EPOCA} episodios = {NUM_EPOCAS * EPISODIOS_POR_EPOCA} episodios totales"
    )
    print(f"Mapas únicos: {NUM_EPOCAS * 12} escenarios diferentes")
    print(
        f"Tiempo estimado: ~6 min/época = ~{NUM_EPOCAS * 6 // 60}h {NUM_EPOCAS * 6 % 60}min"
    )
    print(f"Checkpoints cada 10 épocas ({10 * 6} min)")
    print(f"Sistema de combate y objetos activo")
    print(f"Requisitos de oro por nivel activos")
    print(f"Objetivo: Completar mapa 12 con mejor caña")
    print("=" * 70)

    input("\nPresiona ENTER para comenzar el entrenamiento...")

    # Entrenar
    agente_entrenado, stats = entrenar_por_epocas(
        num_epocas=NUM_EPOCAS, episodios_por_epoca=EPISODIOS_POR_EPOCA
    )

    print("\nEntrenamiento completado exitosamente!")
    print(f"Archivos guardados:")
    print(f"   - agente_entrenado_epocas.pkl")
    print(f"   - qtables_epocas.json")
    print(f"   - mapas_entrenamiento/epoca_XXXX/ (escenarios generados)")
