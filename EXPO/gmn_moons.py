# Ejemplo del uso de GMN (Gaussian Mixture Models) en Python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Generemos datos de ejemplo con MOONS
from sklearn.datasets import make_moons

# Generar datos de ejemplo
n_samples = 300
random_state = 42

X, y_true = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)

gmn = GaussianMixture(n_components=2, random_state=random_state)
gmn.fit(X)

# Predicción probabilística
labels = gmn.predict(X)
probs = gmn.predict_proba(X)
print(probs[:10])

# Después del fit y las predicciones, añadir:
# Calcular métricas de clustering
silhouette = silhouette_score(X, labels)
calinski = calinski_harabasz_score(X, labels)
davies = davies_bouldin_score(X, labels)

print("\n\nMétricas de evaluación del clustering:")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Score: {calinski:.3f}")
print(f"Davies-Bouldin Score: {davies:.3f}")

# Comparemos las predicciones contra las etiquetas verdaderas
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title("Etiquetas verdaderas")

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Clustering con Gaussian Mixture Model")

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=probs[:, 0], cmap='coolwarm')
plt.title("Responsibility of Component 0")
plt.colorbar(label='P(Component 0 | X)')

plt.show()
