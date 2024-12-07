import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(100)

# GBM parameters
r = 2  # Drift (rate of return)
sigma = 1  # Volatility
S0 = 1  # Initial value
T = 1  # Time horizon


# Function to perform Euler-Maruyama for a given dt
def euler_maruyama(dt, S0, r, sigma, T):
    N = int(T / dt)
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)

    St_true = S0 * np.exp((r - 0.5 * sigma**2) * np.linspace(dt, T, N) + sigma * W)

    St_em = S0
    for i in range(N):
        Winc = dW[i]
        St_em += dt * r * St_em + sigma * St_em * Winc

    return St_true[-1], St_em


# Compute average error for different time steps
def compute_average_error(dt, S0, r, sigma, T, num_runs):
    errors = []
    for _ in range(num_runs):
        St_true, St_em = euler_maruyama(dt, S0, r, sigma, T)
        errors.append(abs(St_true - St_em))
    return np.mean(errors)


# Different time steps and number of runs
time_steps = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
num_runs = 1000

# Store results
results = {}
for dt in time_steps:
    avg_error = compute_average_error(dt, S0, r, sigma, T, num_runs)
    results[dt] = avg_error
    print(f"dt = {dt}, Average Error = {avg_error}")

# Plot results
plt.plot(results.keys(), results.values(), marker="o")
plt.xlabel("Time Step (dt)", fontsize=12)
plt.ylabel("Average Error", fontsize=12)
plt.title("Average Error vs Time Step", fontsize=14)
plt.grid(True)
# plt.show()
plt.savefig("extgbm.png")
