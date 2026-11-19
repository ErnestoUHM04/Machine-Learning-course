# Hernández Martínez Ernesto Ulises
# Práctica 7: "Métodos de Validación"

from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# CARGA DE DATASETS
# ===============================
datasets = {
    'Iris': load_iris(),
    'Dígitos': load_digits(),
    'Vino': load_wine()
}

# Clasificador base
KNN = KNeighborsClassifier(n_neighbors=3)

# ===============================
# FUNCIÓN PARA IMPRIMIR MÉTRICAS
# ===============================
def print_metrics(y_true, y_pred):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("-" * 40)

# ===============================
# FUNCIÓN PARA GRAFICAR MÉTRICAS
# ===============================
def plot_metrics(metrics_dict, title):
    datasets_list = list(metrics_dict.keys())
    metric_names = ['f1_score', 'accuracy', 'precision', 'recall']

    fig, axs = plt.subplots(1, len(datasets_list), figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    for i, dataset in enumerate(datasets_list):
        values = [metrics_dict[dataset][m] for m in metric_names]
        axs[i].bar(metric_names, values, color='skyblue')
        axs[i].set_ylim(0, 1)
        axs[i].set_title(dataset.capitalize())
        axs[i].set_ylabel("Score")
        axs[i].set_xticklabels(metric_names, rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ===============================
# CONFIGURAR MÉTRICAS PARA VALIDACIÓN CRUZADA
# ===============================
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

# ===============================
# DICCIONARIOS PARA ALMACENAR RESULTADOS
# ===============================
holdout_metrics = {}
kfold_metrics = {}
stratified_metrics = {}
loo_metrics = {}

# ===============================
# MÉTODO 1: HOLD-OUT
# ===============================
print("🔹 MÉTODO 1: Hold-Out (70% train, 30% test)\n")
for name, data in datasets.items():
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)

    # Mostrar métricas en terminal
    print(f"📊 Dataset: {name}")
    print_metrics(y_test, y_pred)

    # Guardar métricas para graficar
    holdout_metrics[name] = {
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro')
    }

# ===============================
# MÉTODO 2: K-FOLD
# ===============================
print("\n🔹 MÉTODO 2: K-Fold (k=5)\n")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, data in datasets.items():
    X, y = data.data, data.target
    kfold_metrics[name] = {}
    print(f"📊 Dataset: {name}")
    for metric_name, scorer in scoring_metrics.items():
        scores = cross_val_score(KNN, X, y, cv=kf, scoring=scorer)
        mean_score = np.mean(scores)
        # Mostrar métricas en terminal
        print(f"{metric_name.capitalize():<10}: {mean_score:.4f}")
        # Guardar métricas para graficar
        kfold_metrics[name][metric_name if metric_name != 'f1' else 'f1_score'] = mean_score
    print("-" * 40)

# ===============================
# MÉTODO 3: STRATIFIED K-FOLD
# ===============================
print("\n🔹 MÉTODO 3: Stratified K-Fold (k=5)\n")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, data in datasets.items():
    X, y = data.data, data.target
    stratified_metrics[name] = {}
    print(f"📊 Dataset: {name}")
    for metric_name, scorer in scoring_metrics.items():
        scores = cross_val_score(KNN, X, y, cv=skf, scoring=scorer)
        mean_score = np.mean(scores)
        print(f"{metric_name.capitalize():<10}: {mean_score:.4f}")
        stratified_metrics[name][metric_name if metric_name != 'f1' else 'f1_score'] = mean_score
    print("-" * 40)

# ===============================
# MÉTODO 4: LEAVE-ONE-OUT (solo Accuracy)
# ===============================
print("\n🔹 MÉTODO 4: Leave-One-Out (LOO)\n")
loo = LeaveOneOut()
for name, data in datasets.items():
    X, y = data.data, data.target
    scores = cross_val_score(KNN, X, y, cv=loo, scoring='accuracy')
    mean_score = np.mean(scores)
    # Mostrar en terminal
    print(f"📊 Dataset: {name} | Accuracy promedio: {mean_score:.4f}")
    # Guardar para graficar (las demás métricas 0 solo para compatibilidad)
    loo_metrics[name] = {'accuracy': mean_score,
                         'f1_score':0, 'precision':0, 'recall':0}
print("-" * 40)
# ===============================
# GRAFICAR RESULTADOS
# ===============================
plot_metrics(holdout_metrics, "Desempeño KNN - Hold-Out")
plot_metrics(kfold_metrics, "Desempeño KNN - K-Fold")
plot_metrics(stratified_metrics, "Desempeño KNN - Stratified K-Fold")
plot_metrics(loo_metrics, "Desempeño KNN - Leave-One-Out")