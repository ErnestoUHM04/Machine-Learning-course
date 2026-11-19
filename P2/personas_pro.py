# Este es el mero bueno dice Yuko
import random as rd

def creaPoblacion(tamPobla,n):
    poblacion=[]
    for i in range(0,tamPobla):
        individuo=creaIndividuo(n,i)
        poblacion.append(individuo)
    return poblacion

def creaIndividuo(numCromosomas,id):
    individuo=[]
    fitness=0
    for i in range(0,numCromosomas):
        individuo.append(rd.randrange(1,10))
        fitness+=individuo[i]
    individuo.append(id)#penultima casilla es el id para saber si son hermanos
    individuo.append(fitness)#ultima casilla es el fitness
    return individuo

def calculaFitnnessIndv(individuo):
    if(isinstance(individuo,list)):
        fitness=0
        for i in range(0,len(individuo)-2):
            fitness+=individuo[i]
        individuo[-1]=fitness
        return fitness

def calculaFitnessPobla(poblacion):
    for individuo in poblacion:
        calculaFitnnessIndv(individuo)

def mutacion(poblacion,porcentajeMuta):
    if(isinstance(poblacion,list)):
        for individuo in poblacion:
            if(rd.random()<=porcentajeMuta):
                i=rd.randrange(0,len(individuo)-2)
                j=i
                while(j==i):
                    j=rd.randrange(0,len(individuo)-2)
                if(individuo[i]<9):
                    individuo[i]=rd.randrange(1,10)
                if(individuo[j]<9):
                    individuo[j]=rd.randrange(1,10)
            calculaFitnnessIndv(individuo)
        return poblacion

def cruzaPoblacion(poblacion,porcentajeCruza):
    if(isinstance(poblacion,list)):
        poblacion=ordena1ColDesc(poblacion,-2,False)#ordeno por id
        SetA=[]
        SetB=[]
        #indice es una variable axuliar y ya
        indice=0
        while(len(poblacion)>0):
            elemn=poblacion.pop(0)
            if(indice%2==0):
                SetA.append(elemn)
            else:
                SetB.append(elemn)
            indice+=1
        indice=0
        indice,hijosA=cruzaSubconjunto(SetA,indice)
        indice,hijosB=cruzaSubconjunto(SetB,indice)
        while(len(hijosA)>0 or len(hijosB)>0):
            poblacion.append(hijosA.pop(0))
            poblacion.append(hijosB.pop(0))
        poblacion=ordena1ColDesc(poblacion,-1,True)#ordeno por fitness
        return poblacion

def cruzaSubconjunto(SetX,indice):
    if(isinstance(SetX,list)):
        if(len(SetX)%2!=0):
            ValueError("El tam debe ser par")
        SetX=ordena1ColDesc(SetX,-1,True)#ordeno por fitness
        SetHijos=[]
        while(len(SetX)>0):
            indice+=1
            padre1=SetX.pop(0)
            padre2=SetX.pop(0)
            hijo1,hijo2=cruzaIndividuo(padre1,padre2,indice)
            SetHijos.append(hijo1)
            SetHijos.append(hijo2)
        return indice,SetHijos

def cruzaIndividuo(padre1,padre2,id):
    if(isinstance(padre1,list) and isinstance(padre2,list)):
        if(padre1[-2]==padre2[-2]):
            ValueError("Los individuos son hermanos")
        hijo1=[]
        hijo2=[]
        tam=len(padre1)-2#El 2 es porque hay una col fitness y un id
        for i in range(0,tam):
            if(padre1[i]==9 or padre2[i]==9):
                hijo1.append(padre1[i])
                hijo2.append(padre2[i])
            else:
                prom=padre1[i]+padre2[i]
                if(prom%2==0):
                    prom//=2
                    hijo1.append(prom)
                    hijo2.append(prom)
                else:
                    prom=(prom-1)//2#Lo hago par y me voy a su infimo
                    # hijo1.append(prom+1)#Un hijo se asegura de siempre tener el valor sup
                    # hijo2.append(prom)#El otro hijo se asegura de tener el valor inf
                    #El diablo esta en los detalles
                    if(rd.randint(1,2)==2):
                        hijo1.append(prom+1)#Un hijo se asegura de siempre tener el valor sup
                        hijo2.append(prom)#El otro hijo se asegura de tener el valor inf
                    else:
                        hijo1.append(prom)#El otro hijo se asegura de tener el valor inf
                        hijo2.append(prom+1)#Un hijo se asegura de siempre tener el valor sup
        hijo1.append(id)#Son hermanos
        hijo2.append(id)#Son hermanos
        hijo1.append(0)#Fitness
        hijo2.append(0)#Fitness
        calculaFitnnessIndv(hijo1)
        calculaFitnnessIndv(hijo2)
        return hijo1,hijo2

def ordena1ColDesc(matriz, columna,desc_col):
    """
    Ordena la matriz completa por una columna específica en orden descendente
    """
    return sorted(matriz, key=lambda fila: fila[columna], reverse=desc_col)

def verifica(poblacion,numCromosomas,gen):
    if(isinstance(poblacion,list)):
        for individuo in poblacion:
            if(individuo[-1]==9*numCromosomas):
                print(f"EL papuh perfecto aparecio en la generacion={gen}")
                print(individuo)
                return 1
        return 0

def imprimePobla(poblacion):
    print("La poblacion Actual es:")
    for individuo in poblacion:
        print(individuo)

def main():
    semilla=8166
    rd.seed(semilla)
    tamPobla=100
    numCromosomas=20
    porcentajeCruza=0.1
    porcentajeMutacion=0.1
    genmax=2000
    gen=0
    poblacion=creaPoblacion(tamPobla,numCromosomas)
    while(gen<genmax):
        gen+=1
        print(f"La generacion actual es: {gen}")
        poblacion=mutacion(poblacion,porcentajeMutacion)
        poblacion=cruzaPoblacion(poblacion,porcentajeCruza)
        imprimePobla(poblacion)
        if(verifica(poblacion,numCromosomas,gen)==1):
            print(f"seed={semilla}")
            break
    return 0

if __name__=="__main__":
    main()