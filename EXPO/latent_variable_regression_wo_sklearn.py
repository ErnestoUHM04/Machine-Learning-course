import numpy as np
import matplotlib.pyplot as plt

np.random.seed(66)

# Two latent linear functions
n = 300
x1 = np.random.uniform(0, 10, n//2)
y1 = 2*x1 + 1 + np.random.normal(0, 1, n//2)     # line 1

x2 = np.random.uniform(0, 10, n//2)
y2 = -1*x2 + 12 + np.random.normal(0, 1, n//2)    # line 2

X = np.concatenate([x1, x2])
Y = np.concatenate([y1, y2])
plt.scatter(X, Y, s=20)
plt.title("Data from two latent regressions")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


K = 2
n = len(X)

# Initialize weights and parameters
weights = np.ones(K) / K
means = np.random.randn(K)         # slopes (β)
intercepts = np.random.randn(K)    # intercepts
variances = np.ones(K)             # noise variances


def gaussian_pdf(y, mean, var):
    return np.exp(-0.5 * (y - mean)**2 / var) / np.sqrt(2 * np.pi * var)

def E_step(X, Y, weights, means, intercepts, variances):
    N = len(X)
    K = len(weights)
    resp = np.zeros((N, K))
    
    for k in range(K):
        mu = means[k]*X + intercepts[k]
        resp[:, k] = weights[k] * gaussian_pdf(Y, mu, variances[k])
    
    # Normalize
    resp = resp / resp.sum(axis=1, keepdims=True)
    return resp


def M_step(X, Y, resp):
    N, K = resp.shape
    weights = resp.sum(axis=0) / N
    
    means = np.zeros(K)
    intercepts = np.zeros(K)
    variances = np.zeros(K)
    
    for k in range(K):
        r = resp[:, k]
        W = np.diag(r)
        X_aug = np.vstack([X, np.ones_like(X)]).T
        
        # Weighted least squares (β, b)
        beta = np.linalg.inv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ Y
        means[k], intercepts[k] = beta
        
        # Weighted variance
        y_pred = X*means[k] + intercepts[k]
        variances[k] = np.sum(r * (Y - y_pred)**2) / r.sum()
    
    return weights, means, intercepts, variances

for iteration in range(20):
    # E-step
    resp = E_step(X, Y, weights, means, intercepts, variances)
    
    # M-step
    weights, means, intercepts, variances = M_step(X, Y, resp)

print("Final parameters:")
for k in range(K):
    print(f"Component {k}: slope={means[k]:.3f}, intercept={intercepts[k]:.3f}, variance={variances[k]:.3f}, weight={weights[k]:.3f}")

labels = np.argmax(resp, axis=1)

plt.scatter(X, Y, c=labels, cmap='viridis', s=25)
for k in range(K):
    x_line = np.linspace(0, 10, 100)
    y_line = means[k]*x_line + intercepts[k]
    plt.plot(x_line, y_line, label=f"Comp {k}: y={means[k]:.2f}x+{intercepts[k]:.2f}")
plt.legend()
plt.title("EM for Mixture of Linear Regressions")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()