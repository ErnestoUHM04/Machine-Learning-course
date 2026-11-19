import numpy as np
import random
import os
import time
import sys
from collections import deque

class MazeEnv:
    def __init__(self, mode, map_source, filename=None):
        self.size = 10
        self.mode = mode 
        self.map = []
        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None
        self.treasures = []
        self.pits = []
        self.walls = []
        self.collected_treasures = set()
        self.steps = 0
        
        if map_source == 'file':
            self.load_from_file(filename)
        else:
            self.generate_random_map()

    def generate_random_map(self):
        while True:
            self.map = [[' ' for _ in range(self.size)] for _ in range(self.size)]
            self.walls = []
            self.pits = []
            self.treasures = []

            for r in range(self.size):
                for c in range(self.size):
                    if r == 0 or r == self.size - 1 or c == 0 or c == self.size - 1:
                        self.map[r][c] = '#'
                        self.walls.append((r, c))
                    elif random.random() < 0.25: 
                        self.map[r][c] = '#'
                        self.walls.append((r, c))

            self.start_pos = (1, 1)
            self.goal_pos = (self.size - 2, self.size - 2)
            self.map[1][1] = 'S'
            self.map[self.size - 2][self.size - 2] = 'G'
            
            if (1, 1) in self.walls: self.walls.remove((1, 1))
            if (self.size-2, self.size-2) in self.walls: self.walls.remove((self.size-2, self.size-2))

            if self.mode == 2:
                num_treasures = random.randint(1, 3)
                while len(self.treasures) < num_treasures:
                    r, c = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
                    if (r, c) != self.start_pos and (r, c) != self.goal_pos and self.map[r][c] == ' ':
                        self.treasures.append((r, c))
                        self.map[r][c] = 'T'

                num_pits = random.randint(2, 5)
                while len(self.pits) < num_pits:
                    r, c = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
                    if (r, c) != self.start_pos and (r, c) != self.goal_pos and self.map[r][c] == ' ':
                        self.pits.append((r, c))
                        self.map[r][c] = 'X'
            else:
                num_obstacles = random.randint(5, 10)
                while len(self.pits) < num_obstacles:
                    r, c = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
                    if (r, c) != self.start_pos and (r, c) != self.goal_pos and self.map[r][c] == ' ':
                        self.pits.append((r, c))
                        self.map[r][c] = 'X'

            if self.is_solvable():
                break

    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            self.map = [list(line.strip('\n')) for line in lines]
            self.size = len(self.map)
            self.walls = []
            self.pits = []
            self.treasures = []
            
            for r in range(self.size):
                for c in range(len(self.map[r])):
                    char = self.map[r][c]
                    if char == 'S': self.start_pos = (r, c)
                    elif char == 'G': self.goal_pos = (r, c)
                    elif char == 'T': self.treasures.append((r, c))
                    elif char == 'X': self.pits.append((r, c))
                    elif char == '#': self.walls.append((r, c))
                        
            if not self.start_pos or not self.goal_pos:
                raise ValueError("ERROR: Mapa sin inicio (S) o meta (G)")
                
        except Exception as e:
            print(f"Error: {e}")
            exit()

    def is_solvable(self):
        queue = deque([self.start_pos])
        visited = {self.start_pos}
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal_pos:
                return True
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in visited:
                        cell = self.map[nr][nc]
                        if cell == '#':
                            continue
                        if self.mode == 2 and cell == 'X':
                            continue 
                        if self.mode == 1 and cell == 'X':
                             continue
                        
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def reset(self):
        self.agent_pos = self.start_pos
        self.collected_treasures = set()
        self.steps = 0
        return self.agent_pos

    def step(self, action):
        self.steps += 1
        row, col = self.agent_pos
        
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        new_r, new_c = row + dr, col + dc
        
        reward = -1 
        done = False
        new_pos = (row, col)

        if not (0 <= new_r < self.size and 0 <= new_c < self.size):
            reward = -10
            done = True 
        else:
            cell = self.map[new_r][new_c]
            
            if cell == '#':
                reward = -10
                done = True 
            elif cell == 'X':
                reward = -5 if self.mode == 1 else -20
                done = True 
            else:
                new_pos = (new_r, new_c)
                
                if self.mode == 1:
                    if cell == 'G':
                        reward = 50
                        done = True
                
                elif self.mode == 2:
                    if cell == 'T' and new_pos not in self.collected_treasures:
                        reward = 20
                        self.collected_treasures.add(new_pos)
                    elif cell == 'G':
                        if self.steps <= 40: reward += 30
                        if self.steps <= 70: reward += 30
                        if self.steps <= 100: reward += 30
                        done = True

        self.agent_pos = new_pos
        return new_pos, reward, done

    def render(self, delay=0.1, episode_num=None):
        sys.stdout.write("\033[H") 
        output = ""
        if episode_num is not None:
            output += f"--- ENTRENANDO: ITERACION {episode_num} ---\n"
        else:
            output += "--- SOLUCION FINAL ---\n"
            
        output += f"Pasos: {self.steps}\n"
        for r in range(self.size):
            row_str = ""
            for c in range(len(self.map[r])):
                if (r, c) == self.agent_pos:
                    row_str += "A " 
                elif (r, c) in self.collected_treasures and self.map[r][c] == 'T':
                    row_str += "  "
                else:
                    row_str += self.map[r][c] + " "
            output += row_str + "\n"
        
        sys.stdout.write(output)
        sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)

def get_user_config():
    os.system('clear')
    print("--- Configuración Q-Learning ---")
    print("1. Laberinto Estocástico")
    print("2. Laberinto con Tesoros")
    mode = int(input("Selecciona modo (1/2): "))
    
    print("\nGeneración del Mapa:")
    print("1. Generar aleatorio (Dificil)")
    print("2. Cargar desde .txt")
    source_choice = input("Selecciona opción (1/2): ")
    
    filename = None
    if source_choice == '2':
        filename = input("Nombre del archivo (ej. mapa.txt): ")
        source = 'file'
    else:
        source = 'random'
        
    return mode, source, filename

def train_and_animate():
    mode, source, filename = get_user_config()
    env = MazeEnv(mode, source, filename)
    
    q_table = np.zeros((env.size, env.size, 4))
    alpha = 0.1
    gamma = 0.95 
    epsilon = 1.0
    epsilon_decay = 0.999 
    min_epsilon = 0.01
    episodes = 5000 

    os.system('clear')
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            if episode % 100 == 0: 
                env.render(delay=0.0001, episode_num=episode+1)
            
            r, c = state
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[r, c])
            
            next_state, reward, done = env.step(action)
            nr, nc = next_state
            
            old_value = q_table[r, c, action]
            next_max = np.max(q_table[nr, nc])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[r, c, action] = new_value
            state = next_state
            
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    print("\nEntrenamiento completado. Presiona Enter para ver la solución...")
    input()
    
    state = env.reset()
    done = False
    total_reward = 0
    os.system('clear')
    
    while not done:
        env.render(delay=0.4) 
        r, c = state
        action = np.argmax(q_table[r, c])
        state, reward, done = env.step(action)
        total_reward += reward
        
        if env.steps > 100: 
            print("\nEl agente tardó demasiado o entró en bucle.")
            break
    
    env.render(delay=0)
    print(f"\n¡Terminado!")
    print(f"Puntaje Total: {total_reward}")

if __name__ == "__main__":
    train_and_animate()