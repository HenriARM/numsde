import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42)

# Parameters for the Ornstein-Uhlenbeck process
theta = 0.5  # Rate of mean reversion
mu = 0.0  # Long-term mean
sigma = 0.3  # Volatility
X0 = 0.0  # Initial value

# Simulation parameters
T = 2.0  # Total time
N = 1000  # Number of steps
dt = T / N  # Time step

t = np.linspace(0, T, N)  # Time array

# Brownian increments and path
dW = np.sqrt(dt) * np.random.randn(N)  # Brownian increments
W = np.cumsum(dW)  # Cumulative Brownian motion


# Analytical solution for Ornstein-Uhlenbeck process
def ornstein_uhlenbeck_analytical(X0, t, theta, mu, sigma):
    X_analytical = (
        X0 * np.exp(-theta * t)
        + mu * (1 - np.exp(-theta * t))
        + (sigma / np.sqrt(2 * theta))
        * (W * np.exp(-theta * t) - np.exp(-2 * theta * t))
    )
    return X_analytical


# Euler-Maruyama method
def euler_maruyama(X0, dW, dt, theta, sigma):
    X_em = np.zeros(N)
    X_em[0] = X0
    for i in range(1, N):
        drift = -theta * X_em[i - 1] * dt
        diffusion = sigma * dW[i - 1]
        X_em[i] = X_em[i - 1] + drift + diffusion
    return X_em


# Milstein method
def milstein(X0, dW, dt, theta, sigma):
    X_milstein = np.zeros(N)
    X_milstein[0] = X0
    for i in range(1, N):
        drift = -theta * X_milstein[i - 1] * dt
        diffusion = sigma * dW[i - 1]
        diffusion_correction = 0.5 * sigma**2 * (dW[i - 1] ** 2 - dt)
        X_milstein[i] = X_milstein[i - 1] + drift + diffusion + diffusion_correction
    return X_milstein


# Simulate with all methods
X_analytical = ornstein_uhlenbeck_analytical(X0, t, theta, mu, sigma)
X_em = euler_maruyama(X0, dW, dt, theta, sigma)
X_milstein = milstein(X0, dW, dt, theta, sigma)

# X_analytical += 0.3

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, X_analytical, label="Analytical", linestyle="--")
plt.plot(t, X_em, label="Euler-Maruyama")
plt.plot(t, X_milstein, label="Milstein")
plt.title("Ornstein-Uhlenbeck Process: Analytical vs Euler-Maruyama vs Milstein")
plt.xlabel("Time")
plt.ylabel("X_t")
plt.legend()
plt.grid()
# plt.show()

# # Plot differences between methods
# plt.figure(figsize=(10, 6))
# plt.plot(
#     t, np.abs(X_analytical - X_em), label="Analytical vs Euler-Maruyama", linestyle="--"
# )
# plt.plot(
#     t, np.abs(X_analytical - X_milstein), label="Analytical vs Milstein", linestyle="--"
# )
# plt.plot(
#     t, np.abs(X_milstein - X_em), label="Milstein vs Euler-Maruyama", linestyle="--"
# )
# plt.title("Absolute Differences Between Methods")
# plt.xlabel("Time")
# plt.ylabel("Absolute Difference")
# plt.yscale("log")  # Use log scale to highlight small differences
# plt.legend()
# plt.grid()
# plt.show()

plt.savefig("ou.png")