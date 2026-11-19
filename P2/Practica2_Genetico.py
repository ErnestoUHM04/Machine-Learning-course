import random
generaciones = 300
mut = 4 #1/n
numcambio = 6 #numeros de parejas a cambiar entre grupos

def crearPoblacion():
    poblacion = []
    persona = []
    for i in range(0,100):
        for _ in range(0,20):
            persona.append(random.randint(1,9))
        persona.append(i) #IDENTIFICADOR
        poblacion.append(persona)
        persona = []
    return poblacion

def reproduccion(pob):
    newgen = []
    grupos, grupoProm = heuristica(pob)
    pareja_id = 0  # ID único para cada pareja

    for i in range(len(grupoProm)):
        mejor = grupoProm[i][0]
        print(f"Mejor promedio del grupo {i+1}: ",mejor)

    for i in range(len(grupos)):
        for k in range(25): #numero de parejas por grupo
            #SELECCION DE PADRES
            p1 = random.choice(grupos[i])
            p2 = random.choice(grupos[i])

            while p1[-1] == p2[-1]:  #evitar hermanos
                p2 = random.choice(grupos[i])
            grupos[i].remove(p1)
            grupos[i].remove(p2)

            #REPRODUCCION
            hijo1, hijo2 = [], []
            #For para abarcar cada cromosoma
            for j in range(0,20):
                prom = p1[j] + p2[j]
                #DEFINICION PARA REPARTIR LOS VALORES DE LOS PADRES
                if prom % 2 == 0:
                    hijo1.append(int(prom/2))
                    hijo2.append(int(prom/2)) 
                else:
                    if random.randint(1,2) == 1:
                        hijo1.append(int((prom/2)+.5))
                        hijo2.append(int((prom/2)-.5))
                    else:
                        hijo1.append(int((prom/2)-.5))
                        hijo2.append(int((prom/2)+.5))
        # ID HERMANOS
            hijo1.append(pareja_id)
            hijo2.append(pareja_id)
            newgen.append(hijo1)
            newgen.append(hijo2)
            pareja_id += 1
    mutacion(newgen)
    print(newgen)
    #random.shuffle(newgen)
    return newgen

def heuristica(pob): #Selecciona a los padres más aptos
    prom = []
    grupos = []
    grupoProm = []
    # Calculamos el promedio de cada cromosoma junto con el cromosoma original
    pobCont = []
    for sub in pob:
        crom = sub[:-1]  # quita el id
        promedio = sum(crom) / len(crom)
        pobCont.append((promedio, sub))  # guarda tupla (promedio, individuo)
        prom.append(promedio)

    pobOrd = sorted(pobCont) #Ordenar la lista con el promedio
    pobOrd.reverse()
    prom.sort()
    prom.reverse()

    for i in range(2):  #2 grupos, cada grupo tiene 50 personas
        inicio = i*50
        fin = inicio + 50
        grupoProm.append(prom[inicio:fin])
        grupo = []
        for valor in pobOrd[inicio:fin]:
            idx = pobCont.index(valor)   # índice en la lista original
            grupo.append(pob[idx])    # añadimos el cromosoma original


        grupos.append(grupo)

    return grupos, grupoProm

def mutacion(pob):
    for i in range(0,100):
        if random.randint(1,mut) == 1:
            crom = random.randint(0,19)
            x = pob[i][crom]
            pob[i][crom] = random.randint(1,9)
            while x == pob[i][crom]:
                pob[i][crom] = random.randint(1,9)
            print(f"DING DING DING, hubo mutaciooon en el individuo {i} en el cromosoma {crom}, cambiandolo de {x} a {pob[i][crom]}",)

def cambiarGrupos(pob, i_gen):
    grupos = 2
    tamaño_grupo = len(pob) // grupos
    # Dividimos la población en grupos
    lista_grupos = [pob[i*tamaño_grupo:(i+1)*tamaño_grupo] for i in range(grupos)]

    for _ in range(numcambio):
        # Selecciona dos grupos diferentes al azar
        g1, g2 = random.sample(range(grupos), 2)
        # Selecciona un individuo aleatorio de cada grupo
        idx1 = random.randint(0, len(lista_grupos[g1]) - 1)
        idx2 = random.randint(0, len(lista_grupos[g2]) - 1)
        # Intercambia los individuos
        lista_grupos[g1][idx1], lista_grupos[g2][idx2] = lista_grupos[g2][idx2], lista_grupos[g1][idx1]

    # Reconstruir la población completa
    nueva_pob = []
    for grupo in lista_grupos:
        nueva_pob.extend(grupo)

    return nueva_pob


if __name__ == "__main__":
    pob = crearPoblacion()

    for i in range(0,generaciones+1):
        print("GENERACION ",i)
        pob = reproduccion(pob)
        pob = cambiarGrupos(pob, i)
        #EN CASO DE QUE LO ENCUENTRE
        for sub in pob:
            if all(x == 9 for x in sub[:-1]):
                print("SEÑOR PERFECTO ENCONTRADO EN LA GENERACION :",i+1)
                exit()
        
