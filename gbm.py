import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(100)

# GBM parameters
r = 2  # Drift (rate of return)
sigma = 1  # Volatility
S0 = 1  # Initial value
T = 1  # Time horizon
N = 2**8  # Number of time steps

dt = T / N

# Brownian increments and path
dW = np.sqrt(dt) * np.random.randn(N)
W = np.cumsum(dW)

# True solution
St_true = S0 * np.exp((r - 0.5 * sigma**2) * np.arange(dt, T + dt, dt) + sigma * W)

# Plot the true solution
plt.plot(
    np.concatenate(([0], np.arange(dt, T + dt, dt))),
    np.concatenate(([S0], St_true)),
    "m-",
    label="True Solution",
)

# Euler-Maruyama parameters
R = 4
Dt = R * dt
L = N // R

# Initialize solution array
St_em = np.zeros(L)
St_temp = S0

for j in range(L):
    Winc = np.sum(dW[R * j : R * (j + 1)])
    St_temp += Dt * r * St_temp + sigma * St_temp * Winc
    St_em[j] = St_temp

# Plot the Euler-Maruyama approximation
plt.plot(
    np.linspace(0, T, L + 1),
    np.concatenate(([S0], St_em)),
    "r--*",
    label="Euler-Maruyama",
)

# Add labels and legend
plt.xlabel("t", fontsize=12)
plt.ylabel("S", fontsize=16, rotation=0, horizontalalignment="right")
plt.legend()
# plt.show()
plt.savefig("gbm.png")

# Compute the error
emerr = abs(St_em[-1] - St_true[-1])
print(f"Error at final time: {emerr}")
