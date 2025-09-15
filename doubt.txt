
import numpy as np
import pickle
from scipy.special import expit

dim = 59   # trace length per arm
n = 600    # total campaigns
m = 10_000  # number of traces per campaign

ts = np.arange(dim)
rng = np.random.default_rng(seed=42)

# correlation kernel & seasonality
kernel = 1.2 * np.exp(-np.abs(ts[:, None] - ts) / 8.0)
chol = np.linalg.cholesky(kernel)
seasonality = -np.abs(np.sin(np.pi * np.arange(dim) / 7))

data = np.zeros((n, m, dim))
for i in range(n):
    alpha = expit(0.1 * rng.normal() - 0.4)
    k = rng.uniform(0.7, 1.5)
    rates = -0.5 + 0.1 * chol @ rng.normal(size=dim)
    probs = expit(rates + rng.uniform(0.1, 1.0) * seasonality)
    traces = rng.binomial(n=1, p=probs, size=(m, dim))
    max_day = rng.geometric(rng.beta(k * alpha, k * (1 - alpha), size=m)) - 1
    traces[max_day[:, None] <= ts] = 0
    data[i] = traces

with open("synthetic-data-train.pkl", "wb") as f:
    pickle.dump({
        f"campaign-{i:03d}": data[i].astype(bool)
        for i in range(0, 200)
    }, f)

with open("synthetic-data-eval.pkl", "wb") as f:
    pickle.dump({
        f"campaign-{i:03d}": data[i].astype(bool)
        for i in range(200, n)
    }, f)

# Reload and print sample
with open("synthetic-data-train.pkl", "rb") as f:
    train_data_loaded = pickle.load(f)
