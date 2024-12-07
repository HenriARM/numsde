import numpy as np
import matplotlib.pyplot as plt

def set_random_seed(seed=100):
    np.random.seed(seed)

def gbm_parameters():
    r = 2  # Drift (rate of return)
    sigma = 1  # Volatility
    S0 = 1  # Initial value
    T = 1  # Time horizon
    N = 2**8  # Number of time steps
    return r, sigma, S0, T, N

def brownian_motion(N, dt):
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    return dW, W

def true_solution(S0, r, sigma, T, N, W):
    dt = T / N
    St_true = S0 * np.exp((r - 0.5 * sigma**2) * np.arange(dt, T + dt, dt) + sigma * W)
    return St_true

def euler_maruyama(S0, r, sigma, T, N, dW):
    dt = T / N
    R = 4
    Dt = R * dt
    L = N // R
    St_em = np.zeros(L)
    St_temp = S0

    for j in range(L):
        Winc = np.sum(dW[R * j : R * (j + 1)])
        St_temp += Dt * r * St_temp + sigma * St_temp * Winc
        St_em[j] = St_temp

    return St_em, L

def plot_solutions(S0, T, N, St_true, St_em, L):
    dt = T / N
    plt.plot(
        np.concatenate(([0], np.arange(dt, T + dt, dt))),
        np.concatenate(([S0], St_true)),
        "m-",
        label="True Solution",
    )
    plt.plot(
        np.linspace(0, T, L + 1),
        np.concatenate(([S0], St_em)),
        "r--*",
        label="Euler-Maruyama",
    )
    plt.xlabel("t", fontsize=12)
    plt.ylabel("S", fontsize=16, rotation=0, horizontalalignment="right")
    plt.legend()
    plt.savefig("gbm.png")

def compute_error(St_em, St_true):
    emerr = abs(St_em[-1] - St_true[-1])
    return emerr

def main():
    set_random_seed()
    r, sigma, S0, T, N = gbm_parameters()
    dt = T / N
    dW, W = brownian_motion(N, dt)
    St_true = true_solution(S0, r, sigma, T, N, W)
    St_em, L = euler_maruyama(S0, r, sigma, T, N, dW)
    plot_solutions(S0, T, N, St_true, St_em, L)
    emerr = compute_error(St_em, St_true)
    print(f"Error at final time: {emerr}")

if __name__ == "__main__":
    main()