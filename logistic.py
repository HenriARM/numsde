import numpy as np
import matplotlib.pyplot as plt

# Analytical deterministic solution
def logistic_deterministic(X0, t, r, K):
    return K * X0 * np.exp(r * t) / (K + X0 * (np.exp(r * t) - 1))

# Analytical reference (semi-analytical solution)
def logistic_analytical(X0, t, r, K, sigma, W):
    X_det = logistic_deterministic(X0, t, r, K)  # Deterministic part
    stochastic_term = sigma * np.exp(-r * t / 2) * W  # Approx stochastic correction
    return X_det + stochastic_term


def logistic_sde_solution(t, X0, r, K, sigma, dt, W):
    # Discretize time
    N = 2**8
    times = np.linspace(0, 1, N)  # Time array

    # Compute the analytical solution
    exp_term = np.exp((r - sigma**2 / 2) * times + sigma * W)
    integral_term = np.cumsum(exp_term * dt)  # Integral approximation using Riemann sum
    Xt = exp_term / (X0**-1 + (r / K) * integral_term)
    
    return Xt

# Euler-Maruyama Method
def euler_maruyama(X0, dW, dt):
    X = np.zeros(N)
    X[0] = X0
    for i in range(1, N):
        drift = r * X[i - 1] * (1 - X[i - 1] / K) * dt
        diffusion = sigma * X[i - 1] * dW[i - 1]
        X[i] = X[i - 1] + drift + diffusion
    return X

# Milstein Method
def milstein(X0, dW, dt):
    X = np.zeros(N)
    X[0] = X0
    for i in range(1, N):
        drift = r * X[i - 1] * (1 - X[i - 1] / K) * dt
        diffusion = sigma * X[i - 1] * dW[i - 1]
        diffusion_correction = (
            0.5 * sigma**2 * X[i - 1] * (1 - X[i - 1] / K) * ((dW[i - 1]) ** 2 - dt)
        )
        X[i] = X[i - 1] + drift + diffusion + diffusion_correction
    return X

if __name__ == "__main__":
    np.random.seed(42)

    # Parameters
    r = 2.0  # Growth rate
    K = 1.0  # Carrying capacity
    sigma = 0.3  # Volatility
    X0 = 0.1  # Initial value
    T = 1  # Total time
    N = 2**8  # Number of steps
    dt = T / N  # Time step
    t = np.linspace(0, T, N)  # Time points

    # Generate Brownian motion
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)  # Cumulative Brownian motion

    # Simulate with all methods
    X_ref = logistic_analytical(X0, t, r, K, sigma, W)
    X_em = euler_maruyama(X0, dW, dt)
    # X_milstein = milstein(X0, dW, dt)
    X_milstein = logistic_sde_solution(t, X0, r, K, sigma, dt, W)

    # Plot results
    plt.figure(figsize=(12, 6))
    # plt.plot(t, X_ref, label="Reference (Analytical)", linestyle="--", linewidth=2)
    plt.plot(t, X_em, label="Euler-Maruyama", alpha=0.7)
    plt.plot(t, X_milstein, label="aNAL", alpha=0.7)
    plt.title("Comparison of Analytical and Numerical Solutions for Logistic SDE")
    plt.xlabel("Time")
    plt.ylabel("X_t")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("logistic.png")