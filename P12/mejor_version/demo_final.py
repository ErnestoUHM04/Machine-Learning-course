from entorno import AjedrezSimple
from agente import AgenteQLearning, AgenteSARSA
from oponentes import OponenteAleatorio, OponenteGreedy
import time
import os
import platform
import glob

def limpiar_pantalla():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def buscar_agente(caso):
    """
    Busca el archivo de agente disponible para el caso dado.
    Prioridad: _FINAL > _QLEARNING > _SARSA
    """
    opciones = [
        f'agente_caso_{caso}_FINAL.pkl',
        f'agente_caso_{caso}_QLEARNING.pkl',
        f'agente_caso_{caso}_SARSA.pkl',
    ]
    
    for archivo in opciones:
        if os.path.exists(archivo):
            return archivo
    
    # Si no encuentra ninguno, buscar cualquier archivo que empiece con agente_caso_X
    patron = f'agente_caso_{caso}*.pkl'
    archivos = glob.glob(patron)
    if archivos:
        return archivos[0]
    
    return None

def cargar_agente(archivo_agente):
    """
    Carga el agente apropiado según el nombre del archivo
    """
    # Detectar si es SARSA o Q-Learning por el nombre
    if 'SARSA' in archivo_agente.upper():
        agente = AgenteSARSA()
    else:
        agente = AgenteQLearning()
    
    try:
        agente.cargar(archivo_agente)
        return agente
    except FileNotFoundError:
        return None

def jugar_partida_demo(caso='A', oponente_tipo='aleatorio', mostrar_tablero=True, velocidad=1.0):
    if caso == 'A':
        n = 4
        con_peon = False
        nombre_caso = "Caso A: 4x4 (Rey + Torre)"
    else:
        n = 5
        con_peon = True
        nombre_caso = "Caso B: 5x5 (Rey + Torre + Peon)"
    
    # CAMBIO: Buscar archivo automáticamente
    archivo_agente = buscar_agente(caso)
    
    if archivo_agente is None:
        print(f"Error: No se encontró ningún agente entrenado para el Caso {caso}")
        print(f"Archivos buscados:")
        print(f"  - agente_caso_{caso}_FINAL.pkl")
        print(f"  - agente_caso_{caso}_QLEARNING.pkl")
        print(f"  - agente_caso_{caso}_SARSA.pkl")
        return None
    
    # CAMBIO: Cargar el agente apropiado
    agente = cargar_agente(archivo_agente)
    
    if agente is None:
        print(f"Error: No se pudo cargar {archivo_agente}")
        return None
    
    # Mostrar qué agente se está usando
    algoritmo = "SARSA" if isinstance(agente, AgenteSARSA) else "Q-Learning"
    if mostrar_tablero:
        print(f"\n[Usando: {archivo_agente} - {algoritmo}]")
    
    agente.epsilon = 0
    
    if oponente_tipo == 'aleatorio':
        oponente = OponenteAleatorio()
    else:
        oponente = OponenteGreedy()
    
    juego = AjedrezSimple(n=n, con_peon=con_peon)
    estado = juego.reset()
    
    if mostrar_tablero:
        limpiar_pantalla()
    
    print("\n" + "="*60)
    print(f"PARTIDA DE DEMOSTRACION - {nombre_caso}")
    print("="*60)
    print(f"Agente Entrenado ({algoritmo}) vs {oponente.nombre} (Negro)")
    print("="*60)
    
    if mostrar_tablero:
        juego.renderizar()
        time.sleep(velocidad * 2)
    
    movimientos = []
    
    for turno in range(1, 51):
        if mostrar_tablero:
            print(f"\n--- Turno {turno} (Agente) ---")
        
        acciones_agente = juego.obtener_acciones_legales()
        if not acciones_agente:
            if mostrar_tablero:
                print("Agente sin movimientos legales")
            return 'empate', turno, movimientos
        
        accion = agente.elegir_accion(estado, acciones_agente)
        movimientos.append(('agente', accion))
        
        if mostrar_tablero:
            print(f"Agente mueve: {accion[0]} -> {accion[1]}")
        
        estado, recompensa, terminado = juego.hacer_movimiento(accion)
        
        if mostrar_tablero:
            juego.renderizar()
            time.sleep(velocidad)
        
        if terminado:
            if mostrar_tablero:
                print("\n" + "="*60)
                print("AGENTE GANA!")
                print("="*60)
            return 'agente', turno, movimientos
        
        if mostrar_tablero:
            print(f"\n--- Turno {turno} (Oponente) ---")
        
        accion_oponente = oponente.elegir_accion(juego)
        if not accion_oponente:
            if mostrar_tablero:
                print("Oponente sin movimientos legales")
            return 'empate', turno, movimientos
        
        movimientos.append(('oponente', accion_oponente))
        
        if mostrar_tablero:
            print(f"Oponente mueve: {accion_oponente[0]} -> {accion_oponente[1]}")
        
        estado, _, terminado = juego.hacer_movimiento(accion_oponente)
        
        if mostrar_tablero:
            juego.renderizar()
            time.sleep(velocidad)
        
        if terminado:
            if mostrar_tablero:
                print("\n" + "="*60)
                print("OPONENTE GANA!")
                print("="*60)
            return 'oponente', turno, movimientos
    
    if mostrar_tablero:
        print("\n" + "="*60)
        print("EMPATE (maximo de movimientos)")
        print("="*60)
    return 'empate', 50, movimientos

def evaluar_contra_oponente(caso='A', oponente_tipo='aleatorio', num_partidas=100):
    victorias = 0
    derrotas = 0
    empates = 0
    
    print("\n" + "="*60)
    print(f"EVALUACION: {num_partidas} partidas")
    print(f"Caso: {caso} | Oponente: {oponente_tipo}")
    print("="*60)
    
    for i in range(num_partidas):
        resultado, turnos, _ = jugar_partida_demo(caso, oponente_tipo, 
                                                   mostrar_tablero=False, 
                                                   velocidad=0)
        if resultado is None:
            print("Error: No se pudo ejecutar la evaluación")
            return 0, 0, 0
            
        if resultado == 'agente':
            victorias += 1
        elif resultado == 'oponente':
            derrotas += 1
        else:
            empates += 1
        
        if (i + 1) % 20 == 0:
            print(f"Progreso: {i+1}/{num_partidas} partidas")
    
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"Victorias: {victorias} ({victorias/num_partidas*100:.1f}%)")
    print(f"Derrotas:  {derrotas} ({derrotas/num_partidas*100:.1f}%)")
    print(f"Empates:   {empates} ({empates/num_partidas*100:.1f}%)")
    print("="*60 + "\n")
    
    return victorias, derrotas, empates

def menu_demo():
    while True:
        limpiar_pantalla()
        print("\n" + "="*60)
        print("MENU DE DEMOSTRACION - AGENTES FINALES")
        print("="*60)
        
        # Mostrar qué agentes están disponibles
        agente_A = buscar_agente('A')
        agente_B = buscar_agente('B')
        
        print("\nAGENTES DISPONIBLES:")
        if agente_A:
            print(f"  Caso A: {agente_A}")
        else:
            print(f"  Caso A: NO ENCONTRADO")
        
        if agente_B:
            print(f"  Caso B: {agente_B}")
        else:
            print(f"  Caso B: NO ENCONTRADO")
        
        print("\nDEMOS VISUALES:")
        print("  1. Ver partida - Caso A vs Aleatorio")
        print("  2. Ver partida - Caso A vs Greedy")
        print("  3. Ver partida - Caso B vs Aleatorio")
        print("  4. Ver partida - Caso B vs Greedy")
        print("\nEVALUACIONES:")
        print("  5. Evaluar Caso A vs Aleatorio (100 partidas)")
        print("  6. Evaluar Caso A vs Greedy (100 partidas)")
        print("  7. Evaluar Caso B vs Aleatorio (100 partidas)")
        print("  8. Evaluar Caso B vs Greedy (100 partidas)")
        print("\nCOMPARACION COMPLETA:")
        print("  9. Evaluar TODOS los casos")
        print("\nRESUMEN RAPIDO:")
        print("  r. Ver resumen de resultados guardados")
        print("\n  0. Salir")
        print("="*60)
        
        opcion = input("\nSelecciona una opcion: ").strip()
        
        if opcion == '0':
            print("\nAdios!")
            break
        elif opcion == '1':
            jugar_partida_demo('A', 'aleatorio', velocidad=0.8)
            input("\nPresiona Enter para continuar...")
        elif opcion == '2':
            jugar_partida_demo('A', 'greedy', velocidad=0.8)
            input("\nPresiona Enter para continuar...")
        elif opcion == '3':
            jugar_partida_demo('B', 'aleatorio', velocidad=0.8)
            input("\nPresiona Enter para continuar...")
        elif opcion == '4':
            jugar_partida_demo('B', 'greedy', velocidad=0.8)
            input("\nPresiona Enter para continuar...")
        elif opcion == '5':
            evaluar_contra_oponente('A', 'aleatorio', 100)
            input("\nPresiona Enter para continuar...")
        elif opcion == '6':
            evaluar_contra_oponente('A', 'greedy', 100)
            input("\nPresiona Enter para continuar...")
        elif opcion == '7':
            evaluar_contra_oponente('B', 'aleatorio', 100)
            input("\nPresiona Enter para continuar...")
        elif opcion == '8':
            evaluar_contra_oponente('B', 'greedy', 100)
            input("\nPresiona Enter para continuar...")
        elif opcion == '9':
            print("\n" + "="*60)
            print("EVALUACION COMPLETA DE TODOS LOS CASOS")
            print("="*60)
            
            resultados = {}
            
            print("\n[1/4] Caso A vs Aleatorio...")
            resultados['A_aleatorio'] = evaluar_contra_oponente('A', 'aleatorio', 100)
            
            print("\n[2/4] Caso A vs Greedy...")
            resultados['A_greedy'] = evaluar_contra_oponente('A', 'greedy', 100)
            
            print("\n[3/4] Caso B vs Aleatorio...")
            resultados['B_aleatorio'] = evaluar_contra_oponente('B', 'aleatorio', 100)
            
            print("\n[4/4] Caso B vs Greedy...")
            resultados['B_greedy'] = evaluar_contra_oponente('B', 'greedy', 100)
            
            print("\n" + "="*60)
            print("RESUMEN DE RESULTADOS")
            print("="*60)
            print("\nCASO A (4x4):")
            print(f"  vs Aleatorio: {resultados['A_aleatorio'][0]}% victorias")
            print(f"  vs Greedy:    {resultados['A_greedy'][0]}% victorias")
            print("\nCASO B (5x5):")
            print(f"  vs Aleatorio: {resultados['B_aleatorio'][0]}% victorias")
            print(f"  vs Greedy:    {resultados['B_greedy'][0]}% victorias")
            print("="*60)
            
            input("\nPresiona Enter para continuar...")
        elif opcion == 'r' or opcion == 'R':  # ← AGREGAR TODO ESTO
            print("\n" + "="*60)
            print("RESUMEN DE RESULTADOS GUARDADOS")
            print("="*60)
            print("\nCASO A (4x4):")
            print("  vs Aleatorio: 91%")
            print("  vs Greedy:    36%")
            print("  Turnos:       3.1")
            print("\nCASO B (5x5):")
            print("  vs Aleatorio: 98%")
            print("  vs Greedy:    31%")
            print("  Turnos:       9.4")
            print("\nMEJORA POR RL:")
            print("  Caso A: -8.7 puntos vs politica aleatoria")
            print("  Caso B: -8.2 puntos vs politica aleatoria")
            print("="*60)
            input("\nPresiona Enter para continuar...")
        else:
            print("\nOpcion no valida")
            time.sleep(1)

if __name__ == "__main__":
    menu_demo()