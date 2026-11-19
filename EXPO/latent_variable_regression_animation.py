import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(66)

# --- Generate data from two latent regressions ---
n = 300
x1 = np.random.uniform(0, 10, n//2)
y1 = 2*x1 + 1 + np.random.normal(0, 1, n//2)

x2 = np.random.uniform(0, 10, n//2)
y2 = -1*x2 + 12 + np.random.normal(0, 1, n//2)

X = np.concatenate([x1, x2])
Y = np.concatenate([y1, y2])

K = 2
weights = np.ones(K) / K
means = np.random.randn(K)
intercepts = np.random.randn(K)
variances = np.ones(K)


def gaussian_pdf(y, mean, var):
    return np.exp(-0.5 * (y - mean)**2 / var) / np.sqrt(2 * np.pi * var)

def E_step(X, Y, weights, means, intercepts, variances):
    N = len(X)
    K = len(weights)
    resp = np.zeros((N, K))
    
    for k in range(K):
        mu = means[k]*X + intercepts[k]
        resp[:, k] = weights[k] * gaussian_pdf(Y, mu, variances[k])
        
    resp /= resp.sum(axis=1, keepdims=True)
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
        
        beta = np.linalg.inv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ Y
        means[k], intercepts[k] = beta
        
        y_pred = X*means[k] + intercepts[k]
        variances[k] = np.sum(r * (Y - y_pred)**2) / r.sum()
        
    return weights, means, intercepts, variances

fig, ax = plt.subplots()
scat = ax.scatter(X, Y, s=20)
lines = [ax.plot([], [], lw=2)[0] for _ in range(K)]

ax.set_xlim(0, 10)
ax.set_ylim(min(Y)-2, max(Y)+2)
ax.set_title("EM for Mixture of Linear Regressions")
ax.set_xlabel("X")
ax.set_ylabel("Y")

frames_data = []  # to store params per frame

def init():
    for line in lines:
        line.set_data([], [])
    return scat, *lines

def update(frame):
    global weights, means, intercepts, variances
    
    resp = E_step(X, Y, weights, means, intercepts, variances)
    weights, means, intercepts, variances = M_step(X, Y, resp)
    labels = np.argmax(resp, axis=1)
    
    # recolor points by responsibilities
    scat.set_array(labels)
    
    # update regression lines
    x_line = np.linspace(0, 10, 100)
    for k, line in enumerate(lines):
        y_line = means[k]*x_line + intercepts[k]
        line.set_data(x_line, y_line)
    
    ax.set_title(f"Iteration {frame+1}")
    frames_data.append((means.copy(), intercepts.copy()))
    return scat, *lines

anim = FuncAnimation(
    fig, update, init_func=init, frames=20, interval=500, blit=True, repeat=False
)

anim.save('latent_regression_em.gif', writer='pillow', fps=2)
plt.show()


