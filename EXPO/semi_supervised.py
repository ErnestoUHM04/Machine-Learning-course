# ejemplo_semi_supervised_gmm.py
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


# --------------------------
# 1) Generar datos de ejemplo
# --------------------------
# make_blobs crea clusters con centros distintos (útil para ver GMM)
X, y_true = make_blobs(n_samples=400, centers=5, cluster_std=1, random_state=0)
# X: array (400,2) con coordenadas; y_true: etiquetas reales (0,1,2) usadas sólo para generar datos

# --------------------------
# 2) Simular etiquetado parcial
# --------------------------
rng = np.random.RandomState(42)
label_rate = 0.15                    # fracción de puntos etiquetados (15%)
n_labeled = int(label_rate * len(X))
labeled_indices = rng.choice(len(X), size=n_labeled, replace=False)

# Creamos un array de etiquetas parciales: -1 indica "sin etiqueta"
y_partial = np.full(shape=len(X), fill_value=-1, dtype=int)
y_partial[labeled_indices] = y_true[labeled_indices]
# Ahora y_partial contiene etiquetas reales sólo en los índices seleccionados

# --------------------------
# 3) Ajustar GMM a todos los datos (EM)
# --------------------------
n_components = 5  # asumimos conocer (o estimar) el número de componentes
gmm = GaussianMixture(n_components=n_components, random_state=0, covariance_type='full', n_init=5)
gmm.fit(X)  # Aquí corre EM: estima pesos, medias, covarianzas

# --------------------------
# 4) Responsabilidades (E-step resultado): P(component | x)
# --------------------------
probs = gmm.predict_proba(X)  # shape: (n_samples, n_components)
# ogni riga: la distribucion de probabilidades sobre los componentes para ese punto

# --------------------------
# 5) Mapear componentes a etiquetas aprovechando puntos etiquetados
# --------------------------
# Primero, para cada punto etiquetado obtenemos el componente más probable
components_of_labeled = gmm.predict(X[labeled_indices])  # component index para esos puntos
labels_of_labeled = y_partial[labeled_indices]           # etiquetas verdaderas conocidas

# Construímos un mapeo componente -> etiqueta real por mayoría
mapping = {}
for comp in range(n_components):
    # selecciona etiquetas verdaderas de puntos etiquetados que cayeron en este componente
    labels_in_comp = labels_of_labeled[components_of_labeled == comp]
    if len(labels_in_comp) == 0:
        mapping[comp] = -1  # sin información suficiente para este componente
    else:
        # etiqueta mayoritaria
        most_common = Counter(labels_in_comp).most_common(1)[0][0]
        mapping[comp] = most_common

print("Mapa componente -> etiqueta (según puntos etiquetados):", mapping)

# --------------------------
# 6) Asignar etiqueta inferida a TODOS los puntos usando el mapeo
# --------------------------
# componente más probable para cada punto
predicted_components = gmm.predict(X)
# etiqueta inferida según el mapeo (si el componente no tiene mapeo, asignamos -1)
y_inferred = np.array([mapping.get(c, -1) for c in predicted_components])

# --------------------------
# 7) Confianza de la asignación (usamos la prob máxima entre componentes)
# --------------------------
confidences = probs.max(axis=1)  # valor entre 0 y 1; alta = modelo está seguro sobre componente que explica el punto

# 1. Métricas para datos etiquetados
accuracy_labeled = accuracy_score(y_true[labeled_indices], y_inferred[labeled_indices])
f1_labeled = f1_score(y_true[labeled_indices], y_inferred[labeled_indices], average='weighted')

# 2. Métricas de clustering (usando todas las predicciones válidas)
valid_indices = y_inferred != -1
silhouette = silhouette_score(X[valid_indices], y_inferred[valid_indices])

# 3. Métricas de concordancia entre etiquetas verdaderas y predichas
ari = adjusted_rand_score(y_true, y_inferred)
nmi = normalized_mutual_info_score(y_true, y_inferred)

print("\nMétricas de evaluación del aprendizaje semisupervisado:")
print(f"Accuracy en datos etiquetados: {accuracy_labeled:.3f}")
print(f"F1-Score en datos etiquetados: {f1_labeled:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Information: {nmi:.3f}")

# --------------------------
# 8) Plot: etiquetas inferidas + etiquetas reales (parciales)
# --------------------------
plt.figure(figsize=(10, 5))

# a) subplot 1: etiquetas inferidas (colores) y puntos etiquetados destacados
ax1 = plt.subplot(1, 2, 1)
scatter = ax1.scatter(X[:, 0], X[:, 1],
                      c=np.where(y_inferred==-1, -1, y_inferred),  # -1 -> color especial
                      cmap='viridis', s=40, alpha=0.7)
# destacar puntos etiquetados: marcador negro con borde grueso
ax1.scatter(X[labeled_indices, 0], X[labeled_indices, 1],
            facecolors='none', edgecolors='k', s=100, linewidths=1.5, label='Etiquetas conocidas')

ax1.set_title("Etiquetas inferidas por GMM (asociadas via puntos etiquetados)")
ax1.legend()

# b) subplot 2: mismo scatter pero la transparencia refleja "confianza"
ax2 = plt.subplot(1, 2, 2)
"""
# colores según etiqueta inferida (si -1, pintar gris)
colors = np.where(y_inferred==-1, -1, y_inferred)
ax2.scatter(X[:, 0], X[:, 1], c=colors, cmap='viridis', s=40, alpha=confidences)
ax2.scatter(X[labeled_indices, 0], X[labeled_indices, 1],
            facecolors='none', edgecolors='k', s=100, linewidths=1.5)
ax2.set_title("Confianza de la asignación (alpha = prob máxima)")
"""

plt.scatter(X[:, 0], X[:, 1], c=confidences, cmap='coolwarm', s=40)
plt.colorbar(label='Confianza (probabilidad máxima)')
plt.title("Mapa de confianza del GMM")
#plt.show()

plt.tight_layout()
plt.show()