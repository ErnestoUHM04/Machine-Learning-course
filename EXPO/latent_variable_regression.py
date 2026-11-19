import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

# --- 1) Generate synthetic data ---
np.random.seed(42)
n = 300

# Two different linear relationships
x1 = np.random.uniform(0, 10, n//2)
y1 = 2*x1 + 1 + np.random.normal(0, 1, n//2)   # line 1

x2 = np.random.uniform(0, 10, n//2)
y2 = -1*x2 + 12 + np.random.normal(0, 1, n//2)  # line 2

X = np.concatenate([x1, x2])
Y = np.concatenate([y1, y2])

plt.scatter(X, Y, s=20)
plt.title("Data from two hidden regressions")
plt.show()

# Combine into 2D array for GMM
XY = np.column_stack([X, Y])

# Fit a Gaussian Mixture with 2 components
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(XY)

# Predict component (latent cluster) for each point
labels = gmm.predict(XY)

# Plot inferred clusters
plt.scatter(X, Y, c=labels, cmap='viridis', s=30)
plt.title("Clusters inferred by EM (Mixture of Regressions)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


for k in np.unique(labels):
    idx = labels == k
    reg = LinearRegression().fit(X[idx].reshape(-1, 1), Y[idx])
    plt.plot(np.linspace(0, 10, 100),
             reg.predict(np.linspace(0, 10, 100).reshape(-1, 1)),
             label=f'Component {k}: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}')

plt.scatter(X, Y, c=labels, cmap='viridis', s=20)
plt.legend()
plt.title("Regression lines per EM-identified component")
plt.show()