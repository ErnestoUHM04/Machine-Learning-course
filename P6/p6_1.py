# Práctica 6: Reducción de dimensionalidad y agrupación
# Hernández Martínez Ernesto Ulises

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# En cada practica se tiene que:                 IMPORTANTE
# 1. Mostrar la gráfica del dataset sin trabajar o procesar.
# 2. Mostrar las métricas: F1-score, accuracy, precision y recall. Y su matriz de confusión. (con gráficas)
# 3. Mostrar la gráfica del dataset ya trabajado o procesado.

# -------------- PARTE 1: Reducción de dimensionalidad --------------
# Se usarán los datasets:
# - Iris
from sklearn.datasets import load_iris
iris = load_iris()
#print(iris.DESCR)
"""
print("Iris dataset: \n", iris.data[:5], "\n")
plt.figure()
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.title("Iris dataset")
plt.show()
"""
# - Wine Quality
from sklearn.datasets import load_wine
wine = load_wine()
#print(wine.DESCR)
"""
print("Wine dataset: \n", wine.data[:5], "\n")
plt.figure()
plt.scatter(wine.data[:, 0], wine.data[:, 1], c=wine.target)
plt.title("Wine dataset")
plt.show()
"""
# - Digits
from sklearn.datasets import load_digits
digits = load_digits()
#print(digits.DESCR)
"""
print("Digits dataset: \n", digits.data[:5], "\n")
plt.figure()
plt.scatter(digits.data[:, 0], digits.data[:, 1], c=digits.target)
plt.title("Digits dataset")
plt.show()
"""

# Se usarán los siguientes métodos de reducción de dimensionalidad:
# - PCA
from sklearn.decomposition import PCA
# - ICA
from sklearn.decomposition import FastICA
# - LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# - t-SNE
from sklearn.manifold import TSNE
# - UMAP
from umap import UMAP
# - Gaussian Random Projection
from sklearn.random_projection import GaussianRandomProjection

# Herramienta para normalizar los datos:
# - StandardScaler
from sklearn.preprocessing import StandardScaler

# Se usarán los siguientes modelos de clasificación
# - KNN
from sklearn.neighbors import KNeighborsClassifier as KNN
# - SVM
from sklearn.svm import SVC
# - Random Forest
from sklearn.ensemble import RandomForestClassifier
# - Logistic Regression
from sklearn.linear_model import LogisticRegression



###########################################################################
# Lets fucking start
# Function to plot metrics

def plot_metrics(metrics_dict, title):
    datasets = list(metrics_dict.keys())
    metric_names = ['f1_score', 'accuracy', 'precision', 'recall']

    # Create a figure for all datasets
    fig, axs = plt.subplots(1, len(datasets), figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    for i, dataset in enumerate(datasets):
        values = [metrics_dict[dataset][m] for m in metric_names]
        axs[i].bar(metric_names, values, color='skyblue')
        axs[i].set_ylim(0, 1)
        axs[i].set_title(dataset.capitalize())
        axs[i].set_ylabel("Score")
        axs[i].set_xticklabels(metric_names, rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

###########################################################################
# Print digits, wine and iris datasets in 2D using PCA, t-SNE, UMAP and ICA
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
np.random.seed(42)


# Print digits, wine and iris datasets in 2D using PCA
pca = PCA(n_components=2)
iris_2d = pca.fit_transform(iris.data)
wine_2d = pca.fit_transform(wine.data)
digits_2d = pca.fit_transform(digits.data)
# We need the metrics

# PCA Metrics with KNN
pca_metrics_knn = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_2d, iris.target), ('wine', wine_2d, wine.target), ('digits', digits_2d, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = KNN()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pca_metrics_knn[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    pca_metrics_knn[name]['accuracy'] = accuracy_score(y_test, y_pred)
    pca_metrics_knn[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    pca_metrics_knn[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    pca_metrics_knn[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("KNN")
print("PCA Metrics: \n", pca_metrics_knn, "\n")
plot_metrics(pca_metrics_knn, "KNN - PCA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in pca_metrics_knn.keys():
    cm = pca_metrics_knn[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"KNN - {dataset_name.capitalize()} (PCA)")
    plt.show()
"""
# PCA Metrics with SVM
pca_metrics_svm = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_2d, iris.target), ('wine', wine_2d, wine.target), ('digits', digits_2d, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pca_metrics_svm[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    pca_metrics_svm[name]['accuracy'] = accuracy_score(y_test, y_pred)
    pca_metrics_svm[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    pca_metrics_svm[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    pca_metrics_svm[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("SVM")
print("PCA Metrics: \n", pca_metrics_svm, "\n")
plot_metrics(pca_metrics_svm, "SVM - PCA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in pca_metrics_svm.keys():
    cm = pca_metrics_svm[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"SVM - {dataset_name.capitalize()} (PCA)")
    plt.show()
"""
# PCA Metrics with Random Forest
pca_metrics_rf = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_2d, iris.target), ('wine', wine_2d, wine.target), ('digits', digits_2d, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pca_metrics_rf[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    pca_metrics_rf[name]['accuracy'] = accuracy_score(y_test, y_pred)
    pca_metrics_rf[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    pca_metrics_rf[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    pca_metrics_rf[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("Random Forest")
print("PCA Metrics: \n", pca_metrics_rf, "\n")
plot_metrics(pca_metrics_rf, "Random Forest - PCA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in pca_metrics_rf.keys():
    cm = pca_metrics_rf[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Random Forest - {dataset_name.capitalize()} (PCA)")
    plt.show()
"""
# PCA Metrics with Logistic Regression
pca_metrics_lr = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_2d, iris.target), ('wine', wine_2d, wine.target), ('digits', digits_2d, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pca_metrics_lr[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    pca_metrics_lr[name]['accuracy'] = accuracy_score(y_test, y_pred)
    pca_metrics_lr[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    pca_metrics_lr[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    pca_metrics_lr[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("Logistic Regression")
print("PCA Metrics: \n", pca_metrics_lr, "\n")
plot_metrics(pca_metrics_lr, "Logistic Regression - PCA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in pca_metrics_lr.keys():
    cm = pca_metrics_lr[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Logistic Regression - {dataset_name.capitalize()} (PCA)")
    plt.show()
"""

"""
plt.figure()
plt.suptitle("PCA")
plt.subplot(1, 3, 1)
plt.scatter(iris_2d[:, 0], iris_2d[:, 1], c=iris.target)
plt.title("Iris dataset")

plt.subplot(1, 3, 2)
plt.scatter(wine_2d[:, 0], wine_2d[:, 1], c=wine.target)
plt.title("Wine dataset")

plt.subplot(1, 3, 3)
plt.scatter(digits_2d[:, 0], digits_2d[:, 1], c=digits.target)
plt.title("Digits dataset")

plt.show()
"""


# Print digits, wine and iris datasets in 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
iris_tsne = tsne.fit_transform(iris.data)
wine_tsne = tsne.fit_transform(wine.data)
digits_tsne = tsne.fit_transform(digits.data)

# t-SNE Metrics with KNN
tsne_metrics_knn = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_tsne, iris.target), ('wine', wine_tsne, wine.target), ('digits', digits_tsne, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = KNN()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tsne_metrics_knn[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    tsne_metrics_knn[name]['accuracy'] = accuracy_score(y_test, y_pred)
    tsne_metrics_knn[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    tsne_metrics_knn[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    tsne_metrics_knn[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("KNN")
print("t-SNE Metrics: \n", tsne_metrics_knn, "\n")
plot_metrics(tsne_metrics_knn, "KNN - t-SNE Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in tsne_metrics_knn.keys():
    cm = tsne_metrics_knn[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"KNN - {dataset_name.capitalize()} (t-SNE)")
    plt.show()

# t-SNE Metrics with SVM
tsne_metrics_svm = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_tsne, iris.target), ('wine', wine_tsne, wine.target), ('digits', digits_tsne, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tsne_metrics_svm[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    tsne_metrics_svm[name]['accuracy'] = accuracy_score(y_test, y_pred)
    tsne_metrics_svm[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    tsne_metrics_svm[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    tsne_metrics_svm[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("SVM")
print("t-SNE Metrics: \n", tsne_metrics_svm, "\n")
plot_metrics(tsne_metrics_svm, "SVM - t-SNE Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in tsne_metrics_svm.keys():
    cm = tsne_metrics_svm[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"SVM - {dataset_name.capitalize()} (t-SNE)")
    plt.show()
"""
# t-SNE Metrics with Random Forest
tsne_metrics_rf = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_tsne, iris.target), ('wine', wine_tsne, wine.target), ('digits', digits_tsne, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tsne_metrics_rf[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    tsne_metrics_rf[name]['accuracy'] = accuracy_score(y_test, y_pred)
    tsne_metrics_rf[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    tsne_metrics_rf[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    tsne_metrics_rf[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("Random Forest")
print("t-SNE Metrics: \n", tsne_metrics_rf, "\n")
plot_metrics(tsne_metrics_rf, "Random Forest - t-SNE Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in tsne_metrics_rf.keys():
    cm = tsne_metrics_rf[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Random Forest - {dataset_name.capitalize()} (t-SNE)")
    plt.show()
"""
# t-SNE Metrics with Logistic Regression
tsne_metrics_lr = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_tsne, iris.target), ('wine', wine_tsne, wine.target), ('digits', digits_tsne, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tsne_metrics_lr[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    tsne_metrics_lr[name]['accuracy'] = accuracy_score(y_test, y_pred)
    tsne_metrics_lr[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    tsne_metrics_lr[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    tsne_metrics_lr[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("Logistic Regression")
print("t-SNE Metrics: \n", tsne_metrics_lr, "\n")
plot_metrics(tsne_metrics_lr, "Logistic Regression - t-SNE Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in tsne_metrics_lr.keys():
    cm = tsne_metrics_lr[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Logistic Regression - {dataset_name.capitalize()} (t-SNE)")
    plt.show()
"""
"""
plt.figure()
plt.suptitle("t-SNE")
plt.subplot(1, 3, 1)
plt.scatter(iris_tsne[:, 0], iris_tsne[:, 1], c=iris.target)
plt.title("Iris dataset")

plt.subplot(1, 3, 2)
plt.scatter(wine_tsne[:, 0], wine_tsne[:, 1], c=wine.target)
plt.title("Wine dataset")

plt.subplot(1, 3, 3)
plt.scatter(digits_tsne[:, 0], digits_tsne[:, 1], c=digits.target)
plt.title("Digits dataset")

plt.show()
"""


# Print digits, wine and iris datasets in 2D using UMAP
umap_model = UMAP(n_components=2, random_state=42)
iris_umap = umap_model.fit_transform(iris.data)
wine_umap = umap_model.fit_transform(wine.data)
digits_umap = umap_model.fit_transform(digits.data)

# UMAP Metrics with KNN
umap_metrics_knn = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_umap, iris.target), ('wine', wine_umap, wine.target), ('digits', digits_umap, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = KNN()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    umap_metrics_knn[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    umap_metrics_knn[name]['accuracy'] = accuracy_score(y_test, y_pred)
    umap_metrics_knn[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    umap_metrics_knn[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    umap_metrics_knn[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("KNN")
print("UMAP Metrics: \n", umap_metrics_knn, "\n")
plot_metrics(umap_metrics_knn, "KNN - UMAP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in umap_metrics_knn.keys():
    cm = umap_metrics_knn[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"KNN - {dataset_name.capitalize()} (UMAP)")
    plt.show()
"""
# UMAP Metrics with SVM
umap_metrics_svm = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_umap, iris.target), ('wine', wine_umap, wine.target), ('digits', digits_umap, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    umap_metrics_svm[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    umap_metrics_svm[name]['accuracy'] = accuracy_score(y_test, y_pred)
    umap_metrics_svm[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    umap_metrics_svm[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    umap_metrics_svm[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("SVM")
print("UMAP Metrics: \n", umap_metrics_svm, "\n")
plot_metrics(umap_metrics_svm, "SVM - UMAP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in umap_metrics_svm.keys():
    cm = umap_metrics_svm[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"SVM - {dataset_name.capitalize()} (UMAP)")
    plt.show()
"""
# UMAP Metrics with Random Forest
umap_metrics_rf = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_umap, iris.target), ('wine', wine_umap, wine.target), ('digits', digits_umap, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    umap_metrics_rf[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    umap_metrics_rf[name]['accuracy'] = accuracy_score(y_test, y_pred)
    umap_metrics_rf[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    umap_metrics_rf[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    umap_metrics_rf[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("Random Forest")
print("UMAP Metrics: \n", umap_metrics_rf, "\n")
plot_metrics(umap_metrics_rf, "Random Forest - UMAP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in umap_metrics_rf.keys():
    cm = umap_metrics_rf[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Random Forest - {dataset_name.capitalize()} (UMAP)")
    plt.show()
"""
# UMAP Metrics with Logistic Regression
umap_metrics_lr = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_umap, iris.target), ('wine', wine_umap, wine.target), ('digits', digits_umap, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    umap_metrics_lr[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    umap_metrics_lr[name]['accuracy'] = accuracy_score(y_test, y_pred)
    umap_metrics_lr[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    umap_metrics_lr[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    umap_metrics_lr[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)
"""
print("Logistic Regression")
print("UMAP Metrics: \n", umap_metrics_lr, "\n")
plot_metrics(umap_metrics_lr, "Logistic Regression - UMAP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in umap_metrics_lr.keys():
    cm = umap_metrics_lr[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Logistic Regression - {dataset_name.capitalize()} (UMAP)")
    plt.show()
"""

"""
plt.figure()
plt.suptitle("UMAP")
plt.subplot(1, 3, 1)
plt.scatter(iris_umap[:, 0], iris_umap[:, 1], c=iris.target)
plt.title("Iris dataset")

plt.subplot(1, 3, 2)
plt.scatter(wine_umap[:, 0], wine_umap[:, 1], c=wine.target)
plt.title("Wine dataset")

plt.subplot(1, 3, 3)
plt.scatter(digits_umap[:, 0], digits_umap[:, 1], c=digits.target)
plt.title("Digits dataset")

plt.show()
"""


# Print digits, wine and iris datasets in 2D using ICA
ica = FastICA(n_components=2, random_state=42)
iris_ica = ica.fit_transform(iris.data)
wine_ica = ica.fit_transform(wine.data)
digits_ica = ica.fit_transform(digits.data)

# ICA Metrics with KNN
ica_metrics_knn = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_ica, iris.target), ('wine', wine_ica, wine.target), ('digits', digits_ica, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = KNN()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ica_metrics_knn[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    ica_metrics_knn[name]['accuracy'] = accuracy_score(y_test, y_pred)
    ica_metrics_knn[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    ica_metrics_knn[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    ica_metrics_knn[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("KNN")
print("ICA Metrics: \n", ica_metrics_knn, "\n")
plot_metrics(ica_metrics_knn, "KNN - ICA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in ica_metrics_knn.keys():
    cm = ica_metrics_knn[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"KNN - {dataset_name.capitalize()} (ICA)")
    plt.show()

# ICA Metrics with SVM
ica_metrics_svm = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_ica, iris.target), ('wine', wine_ica, wine.target), ('digits', digits_ica, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ica_metrics_svm[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    ica_metrics_svm[name]['accuracy'] = accuracy_score(y_test, y_pred)
    ica_metrics_svm[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    ica_metrics_svm[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    ica_metrics_svm[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("SVM")
print("ICA Metrics: \n", ica_metrics_svm, "\n")
plot_metrics(ica_metrics_svm, "SVM - ICA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in ica_metrics_svm.keys():
    cm = ica_metrics_svm[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"SVM - {dataset_name.capitalize()} (ICA)")
    plt.show()

# ICA Metrics with Random Forest
ica_metrics_rf = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_ica, iris.target), ('wine', wine_ica, wine.target), ('digits', digits_ica, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ica_metrics_rf[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    ica_metrics_rf[name]['accuracy'] = accuracy_score(y_test, y_pred)
    ica_metrics_rf[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    ica_metrics_rf[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    ica_metrics_rf[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("Random Forest")
print("ICA Metrics: \n", ica_metrics_rf, "\n")
plot_metrics(ica_metrics_rf, "Random Forest - ICA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in ica_metrics_rf.keys():
    cm = ica_metrics_rf[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Random Forest - {dataset_name.capitalize()} (ICA)")
    plt.show()

# ICA Metrics with Logistic Regression
ica_metrics_lr = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_ica, iris.target), ('wine', wine_ica, wine.target), ('digits', digits_ica, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ica_metrics_lr[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    ica_metrics_lr[name]['accuracy'] = accuracy_score(y_test, y_pred)
    ica_metrics_lr[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    ica_metrics_lr[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    ica_metrics_lr[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("Logistic Regression")
print("ICA Metrics: \n", ica_metrics_lr, "\n")
plot_metrics(ica_metrics_lr, "Logistic Regression - ICA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in ica_metrics_lr.keys():
    cm = ica_metrics_lr[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Logistic Regression - {dataset_name.capitalize()} (ICA)")
    plt.show()


"""
plt.figure()
plt.suptitle("ICA")
plt.subplot(1, 3, 1)
plt.scatter(iris_ica[:, 0], iris_ica[:, 1], c=iris.target)
plt.title("Iris dataset")

plt.subplot(1, 3, 2)
plt.scatter(wine_ica[:, 0], wine_ica[:, 1], c=wine.target)
plt.title("Wine dataset")

plt.subplot(1, 3, 3)
plt.scatter(digits_ica[:, 0], digits_ica[:, 1], c=digits.target)
plt.title("Digits dataset")

plt.show()
"""

# Print digits, wine and iris datasets in 2D using LDA
lda = LDA(n_components=2)
iris_lda = lda.fit_transform(iris.data, iris.target)
wine_lda = lda.fit_transform(wine.data, wine.target)
digits_lda = lda.fit_transform(digits.data, digits.target)

# LDA Metrics with KNN
lda_metrics_knn = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_lda, iris.target), ('wine', wine_lda, wine.target), ('digits', digits_lda, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = KNN()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lda_metrics_knn[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    lda_metrics_knn[name]['accuracy'] = accuracy_score(y_test, y_pred)
    lda_metrics_knn[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    lda_metrics_knn[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    lda_metrics_knn[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("KNN")
print("LDA Metrics: \n", lda_metrics_knn, "\n")
plot_metrics(lda_metrics_knn, "KNN - LDA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in lda_metrics_knn.keys():
    cm = lda_metrics_knn[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"KNN - {dataset_name.capitalize()} (LDA)")
    plt.show()

# LDA Metrics with SVM
lda_metrics_svm = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_lda, iris.target), ('wine', wine_lda, wine.target), ('digits', digits_lda, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lda_metrics_svm[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    lda_metrics_svm[name]['accuracy'] = accuracy_score(y_test, y_pred)
    lda_metrics_svm[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    lda_metrics_svm[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    lda_metrics_svm[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("SVM")
print("LDA Metrics: \n", lda_metrics_svm, "\n")
plot_metrics(lda_metrics_svm, "SVM - LDA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in lda_metrics_svm.keys():
    cm = lda_metrics_svm[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"SVM - {dataset_name.capitalize()} (LDA)")
    plt.show()

# LDA Metrics with Random Forest
lda_metrics_rf = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_lda, iris.target), ('wine', wine_lda, wine.target), ('digits', digits_lda, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lda_metrics_rf[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    lda_metrics_rf[name]['accuracy'] = accuracy_score(y_test, y_pred)
    lda_metrics_rf[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    lda_metrics_rf[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    lda_metrics_rf[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("Random Forest")
print("LDA Metrics: \n", lda_metrics_rf, "\n")
plot_metrics(lda_metrics_rf, "Random Forest - LDA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in lda_metrics_rf.keys():
    cm = lda_metrics_rf[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Random Forest - {dataset_name.capitalize()} (LDA)")
    plt.show()

# LDA Metrics with Logistic Regression
lda_metrics_lr = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_lda, iris.target), ('wine', wine_lda, wine.target), ('digits', digits_lda, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lda_metrics_lr[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    lda_metrics_lr[name]['accuracy'] = accuracy_score(y_test, y_pred)
    lda_metrics_lr[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    lda_metrics_lr[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    lda_metrics_lr[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("Logistic Regression")
print("LDA Metrics: \n", lda_metrics_lr, "\n")
plot_metrics(lda_metrics_lr, "Logistic Regression - LDA Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in lda_metrics_lr.keys():
    cm = lda_metrics_lr[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Logistic Regression - {dataset_name.capitalize()} (LDA)")
    plt.show()


"""
plt.figure()
plt.suptitle("LDA")
plt.subplot(1, 3, 1)
plt.scatter(iris_lda[:, 0], iris_lda[:, 1], c=iris.target)
plt.title("Iris dataset")

plt.subplot(1, 3, 2)
plt.scatter(wine_lda[:, 0], wine_lda[:, 1], c=wine.target)
plt.title("Wine dataset")

plt.subplot(1, 3, 3)
plt.scatter(digits_lda[:, 0], digits_lda[:, 1], c=digits.target)
plt.title("Digits dataset")

plt.show()
"""

# Print digits, wine and iris datasets in 2D using Gaussian Random Projection
grp = GaussianRandomProjection(n_components=2, random_state=42)
iris_grp = grp.fit_transform(iris.data)
wine_grp = grp.fit_transform(wine.data)
digits_grp = grp.fit_transform(digits.data)

# GRP Metrics with KNN
grp_metrics_knn = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_grp, iris.target), ('wine', wine_grp, wine.target), ('digits', digits_grp, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = KNN()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    grp_metrics_knn[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    grp_metrics_knn[name]['accuracy'] = accuracy_score(y_test, y_pred)
    grp_metrics_knn[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    grp_metrics_knn[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    grp_metrics_knn[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("KNN")
print("GRP Metrics: \n", grp_metrics_knn, "\n")
plot_metrics(grp_metrics_knn, "KNN - GRP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in grp_metrics_knn.keys():
    cm = grp_metrics_knn[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"KNN - {dataset_name.capitalize()} (GRP)")
    plt.show()

# GRP Metrics with SVM
grp_metrics_svm = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_grp, iris.target), ('wine', wine_grp, wine.target), ('digits', digits_grp, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    grp_metrics_svm[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    grp_metrics_svm[name]['accuracy'] = accuracy_score(y_test, y_pred)
    grp_metrics_svm[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    grp_metrics_svm[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    grp_metrics_svm[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("SVM")
print("GRP Metrics: \n", grp_metrics_svm, "\n")
plot_metrics(grp_metrics_svm, "SVM - GRP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in grp_metrics_svm.keys():
    cm = grp_metrics_svm[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"SVM - {dataset_name.capitalize()} (GRP)")
    plt.show()

# GRP Metrics with Random Forest
grp_metrics_rf = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_grp, iris.target), ('wine', wine_grp, wine.target), ('digits', digits_grp, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    grp_metrics_rf[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    grp_metrics_rf[name]['accuracy'] = accuracy_score(y_test, y_pred)
    grp_metrics_rf[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    grp_metrics_rf[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    grp_metrics_rf[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("Random Forest")
print("GRP Metrics: \n", grp_metrics_rf, "\n")
plot_metrics(grp_metrics_rf, "Random Forest - GRP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in grp_metrics_rf.keys():
    cm = grp_metrics_rf[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Random Forest - {dataset_name.capitalize()} (GRP)")
    plt.show()

# GRP Metrics with Logistic Regression
grp_metrics_lr = {'iris': {}, 'wine': {}, 'digits': {}}
for name, data, target in [('iris', iris_grp, iris.target), ('wine', wine_grp, wine.target), ('digits', digits_grp, digits.target)]:
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    grp_metrics_lr[name]['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    grp_metrics_lr[name]['accuracy'] = accuracy_score(y_test, y_pred)
    grp_metrics_lr[name]['precision'] = precision_score(y_test, y_pred, average='weighted')
    grp_metrics_lr[name]['recall'] = recall_score(y_test, y_pred, average='weighted')
    grp_metrics_lr[name]['confusion_matrix'] = confusion_matrix(y_test, y_pred)

print("Logistic Regression")
print("GRP Metrics: \n", grp_metrics_lr, "\n")
plot_metrics(grp_metrics_lr, "Logistic Regression - GRP Metrics")
# Despliega las matrices de confusión para cada dataset
for dataset_name in grp_metrics_lr.keys():
    cm = grp_metrics_lr[dataset_name]['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="BuGn")
    plt.title(f"Logistic Regression - {dataset_name.capitalize()} (GRP)")
    plt.show()


"""
plt.figure()
plt.suptitle("Gaussian Random Projection")
plt.subplot(1, 3, 1)
plt.scatter(iris_grp[:, 0], iris_grp[:, 1], c=iris.target)
plt.title("Iris dataset")

plt.subplot(1, 3, 2)
plt.scatter(wine_grp[:, 0], wine_grp[:, 1], c=wine.target)
plt.title("Wine dataset")

plt.subplot(1, 3, 3)
plt.scatter(digits_grp[:, 0], digits_grp[:, 1], c=digits.target)
plt.title("Digits dataset")

plt.show()
"""