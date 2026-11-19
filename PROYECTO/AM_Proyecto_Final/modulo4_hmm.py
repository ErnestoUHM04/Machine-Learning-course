# MODULO 4 - HIDDEN MARKOV MODELS (HMM)
# Este módulo será usado para determinar los tipos de enemigos que aparecen en un videojuego
# Basado en el clima y el area en la que se encuentra el jugador
# Cabe resaltar que para comunicarnos con los otros módulos, usaremos indices de enteros para representar los estados y observaciones

import numpy as np
from hmmlearn import hmm

# En el siguiente enlace se encuentra la página de documentación de la librería utilizada, léase si se tiene dudas, o para comprender más a fondo el funcionamiento del modelo
# https://hmmlearn.readthedocs.io/en/latest/

# Definición de los estados ocultos   <---- Se refiere a los estados no visibles
# ================================= Diccionario de estados: ===============================
# ESTADO DE ENEMIGOS:                                               NÚMERO REPRESENTATIVO:
# NORMAL: Enemigos comunes.                                         +1
# ENJOADO: Enemigos más agresivos.                                  +3
# ENCABRONADO: Enemigos muy agresivos y difíciles de vencer.        +6
# =========================================================================================

states = ["Normal", "Enjoado", "Encabronado"]
n_states = len(states)
# print('Número de hidden states: ',n_states) # DEBUGG

# Definición de las observaciones    <---- Se refiere a los estados visibles
# El clima, o estado observable, influye en el tipo de enemigos que aparecen, y este está directamente relacionado con el módulo 2
# MODULO 2 - MODELO PROBABILÍSTICO
# Este módulo se relaciona a su vez directamnente con el modulo 3
# MODULO 3 - MAPA Y SERIES DE TIEMPO
# ================================= Diccionario de estados: ===============================
# ESTADO DEL CLIMA:                                               NÚMERO REPRESENTATIVO:
# SOLEADO: Clima despejado y soleado.                               S - 0
# LLUVIA: Clima nublado y lluvioso.                                 L - 1
# TORMENTA: Clima tormentoso y peligroso.                           T - 2
# =========================================================================================
# *Nota: cabe aclarar que aún no conocemos el diccionario de datos de estos, por lo que usaremos +1, +2, +3 por lo mientras

observations = ["Soleado", "Lluvia", "Tormenta"]
n_observations = len(observations)
# print('Número de observaciones: ',n_observations) # DEBUGG

# Para el siguiente paso, se tienen que definir las siguientes matrices:
# PROBABILIDAD DE ESTADOS:                 <--- Esta se refiere a la probabilidad inicial de cada estado oculto
#
# PROBABILIDAD DE TRANSICIONES:            <--- Esta se refiere a la probabilida de pasar entre estados
#
# PROBABILIDAD DE EMISIÓN:                 <--- Esta se refiere a la probabilida de que un estado oculto represente a uno de los estados observados
#

# ================================= Probabilidad inicial de estados =======================
#                  Normal   |   Enojado  |   Encabronado    |
# -------------------------------------------------------------------
#               |    0.3    |     0.5    |       0.2        |
# -------------------------------------------------------------------
# *Nota: cabe aclarar que esta matriz se verá explicada con un pdf que se mandará al grupo
# *Nota: esta matriz fue propuesta por Ernesto, si se desea cambiar, unicamente cuidar la condición de que cada renglón sume 1

state_probability = np.array([0.3, 0.5, 0.2])
# print("\nState probability: \n", state_probability) # DEBUGG

# ================================= Matriz de Transiciones ================================
# Aquí se ingreserá la matriz de transiciones, esta será representada de la siguiente manera:
#                  Normal   |   Enojado  |   Encabronado    |
# -------------------------------------------------------------------
# Normal        |    0.2    |     0.5    |       0.3        |
# Enojado       |    0.3    |     0.35   |       0.35       |
# Encabronado   |    0.15   |     0.35   |       0.5        |
# -------------------------------------------------------------------
# *Nota: cabe aclarar que esta matriz se verá explicada con un pdf que se mandará al grupo
# *Nota: esta matriz fue propuesta por Ernesto, si se desea cambiar, unicamente cuidar la condición de que cada renglón sume 1

transition_probability = np.array([[0.2, 0.5, 0.3],
                                   [0.3, 0.35, 0.35],
                                   [0.15, 0.35, 0.5]])
# print("\nTransition probability: \n", transition_probability) # DEBUGG

# ================================= Matriz de Emisiones ===================================
# Aquí se ingresará la matriz de emisiones, esta será representada de la siguiente manera:
#                  Soleado   |   Lluvia  |   Tormenta       |
# -------------------------------------------------------------------
# Normal        |    0.6    |     0.25   |       0.15       |
# Enojado       |    0.25   |     0.5    |       0.25       |
# Encabronado   |    0.1    |     0.3    |       0.6        |
# -------------------------------------------------------------------
# *Nota: cabe aclarar que esta matriz se verá explicada con un pdf que se mandará al grupo
# *Nota: esta matriz fue propuesta por Ernesto, si se desea cambiar, unicamente cuidar la condición de que cada renglón sume 1

emission_probability= np.array([[0.6, 0.25, 0.15],
                                [0.25, 0.5, 0.25],
                                [0.1, 0.3, 0.6]])
# print("\nEmission probability: \n", emission_probability) # DEBUGG

# Una vez con las matrices necesarias definidas, usamos la librería de hmm para conseguir un modelo que use nuestras matrices para poder generar predicciones
# Se está eligiendo de la libreria el modelo categórico.     <---       CategoricalHMM
# class hmmlearn.hmm.CategoricalHMM(n_components=1, startprob_prior=1.0, transmat_prior=1.0, *, emissionprob_prior=1.0, n_features=None, algorithm='viterbi', random_state=None, n_iter=10, tol=0.01, verbose=False, params='ste', init_params='ste', implementation='log')
#
# Variables:
# - n_features (int) – Number of possible symbols emitted by the model (in the samples).
# - monitor (ConvergenceMonitor) – Monitor object used to check the convergence of EM.
# - startprob (array, shape (n_components, )) – Initial state occupation distribution.
# - transmat (array, shape (n_components, n_components)) – Matrix of transition probabilities between states.
# - emissionprob (array, shape (n_components, n_features)) – Probability of emitting a given symbol when in each state.
#
# Parameters:
# - n_components (int) – Number of states.
# - startprob_prior (array, shape (n_components, ), optional) – Parameters of the Dirichlet prior distribution for startprob_.
# - transmat_prior (array, shape (n_components, n_components), optional) – Parameters of the Dirichlet prior distribution for each row of the transition probabilities transmat_.
# - emissionprob_prior (array, shape (n_components, n_features), optional) – Parameters of the Dirichlet prior distribution for emissionprob_.
# - n_features (int, optional) – The number of categorical symbols in the HMM. Will be inferred from the data if not set.
# - algorithm ({"viterbi", "map"}, optional) – Decoder algorithm.
#           ”viterbi”: finds the most likely sequence of states, given all emissions.
#           ”map” (also known as smoothing or forward-backward): finds the sequence of the individual most-likely states, given all emissions.
# - random_state (RandomState or an int seed, optional) – A random number generator instance.
# - n_iter (int, optional) – Maximum number of iterations to perform.
# - tol (float, optional) – Convergence threshold. EM will stop if the gain in log-likelihood is below this value.
# - verbose (bool, optional) – Whether per-iteration convergence reports are printed to sys.stderr. Convergence can also be diagnosed using the monitor_ attribute.
# - params (string, optional) – The parameters that get updated during (params) or initialized before (init_params) the training. Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, and ‘e’ for emissionprob. Defaults to all parameters.
# - init_params (string, optional) – The parameters that get updated during (params) or initialized before (init_params) the training. Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, and ‘e’ for emissionprob. Defaults to all parameters.
# - implementation (string, optional) – Determines if the forward-backward algorithm is implemented with logarithms (“log”), or using scaling (“scaling”). The default is to use logarithms for backwards compatability.

model = hmm.CategoricalHMM(
    n_components = n_states,
    n_features = n_observations,
    init_params = "",
    )

model.startprob_ = state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

"""
# Print para DEBUGG
print("\n", model.startprob_, "\n")
print(model.transmat_, "\n")
print(model.emissionprob_, "\n")
"""

def predecir_tipo_enemigos(clima_observado, print_log_prob_and_hidden_states = False):
    # Aqui es donde dependemos de lo que mande el otro módulo, y donde lograremos responder con nuestro estado oculto más probable
    # Lo mejor sería recibir inputs de tanto en tanto, para que no se devuelva el mismo estado oculto siempre que se manda el estado observable
    # Habrá que considerar esto para la práctica

    # clima_observado será una lista o array de enteros, que represente una secuencia de climas

    observations_sequence = np.array(clima_observado).reshape(-1, 1)

    # En esta última sección, se consigue log probability y los hidden states

    log_probability, hidden_states = model.decode(
        observations_sequence,
        algorithm='viterbi'
    )

    if print_log_prob_and_hidden_states:
        print('\nLog Probability :',log_probability)
        print("Most likely hidden states:", hidden_states)

        decoded_states = [states[i] for i in hidden_states]
        print("Hidden states (nombres)", decoded_states)

    # Antes de retornar los estados ocultos, se deben de codificar de acuerdo a lo acordado para poder comunicarse adecuadamente con los otros módulos

    # ================================= Diccionario de estados: ===============================
    # ESTADO DE ENEMIGOS:                                               NÚMERO REPRESENTATIVO:
    # NORMAL: Enemigos comunes.                                         +1.       <--->       0
    # ENJOADO: Enemigos más agresivos.                                  +3.       <--->       1
    # ENCABRONADO: Enemigos muy agresivos y difíciles de vencer.        +6.       <--->       2
    # =========================================================================================

    mapped_states = [1 if state == 0 else 3 if state == 1 else 6 for state in hidden_states]

    # De esta manera nos aseguramos de mandar los estados de acuerdo a lo acordado

    return mapped_states

# DEBUGG
print(predecir_tipo_enemigos((1,2,0, 0, 0, 0, 1, 2, 2, 2)))