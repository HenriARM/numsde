import numpy as np
import matplotlib.pyplot as plt


def set_random_seed(seed=100):
    np.random.seed(seed)


def gbm_parameters():
    r = 2  # Drift (rate of return)
    sigma = 1  # Volatility
    S0 = 1  # Initial value
    T = 1  # Time horizon
    return r, sigma, S0, T


def brownian_motion(N, dt):
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    return dW, W


def true_solution(S0, r, sigma, T, dt, W):
    St_true = S0 * np.exp((r - 0.5 * sigma**2) * np.arange(dt, T + dt, dt) + sigma * W)
    return St_true


def euler_maruyama(S0, r, sigma, N, dW, dt):
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


def plot_solutions(S0, T, dt, St_true, St_em, L, color="b"):
    if len(St_true) != 0:
        plt.plot(
            np.concatenate(([0], np.arange(dt, T + dt, dt))),
            np.concatenate(([S0], St_true)),
            "r-",
            label="True Solution",
        )
    plt.plot(
        np.linspace(0, T, L + 1),
        np.concatenate(([S0], St_em)),
        "--*",
        label=f"EM dt={dt:.5f}",
        color=color,
    )
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Stock (S)", fontsize=16, rotation=0, horizontalalignment="right")
    plt.legend()


def compute_error(St_em, St_true):
    emerr = abs(St_em[-1] - St_true[-1])
    return emerr


def main():
    set_random_seed()
    r, sigma, S0, T = gbm_parameters()
    n_values = [2**7, 2**8, 2**9]

    # Calculate final time average error for different values of dt
    num_runs = 1000
    results = {}
    for N in n_values:
        dt = T / N
        errors = []
        for _ in range(num_runs):
            dW, W = brownian_motion(N, dt)
            St_true = true_solution(S0, r, sigma, T, dt, W)
            St_em, _ = euler_maruyama(S0, r, sigma, N, dW, dt)
            errors.append(compute_error(St_em, St_true))
        avg_error = np.mean(errors)
        results[dt] = avg_error
        print(f"dt = {dt}, Average Error = {avg_error}")

    # Plot results
    plt.plot(results.keys(), results.values(), marker="o")
    plt.xlabel("Time Step (dt)", fontsize=12)
    plt.ylabel("Average Error", fontsize=12)
    plt.title("Average Error vs Time Step", fontsize=14)
    plt.grid(True)
    plt.savefig("error_vs_dt.png")

    # Compare EMs with different dt
    N_fix = 2**9
    dt_fix = T / N_fix
    dW_fix, W_fix = brownian_motion(N_fix, dt_fix)
    St_true = true_solution(S0, r, sigma, T, dt_fix, W_fix)
    St_em, L = euler_maruyama(S0, r, sigma, N_fix, dW_fix, dt_fix)
    plt.figure(figsize=(15, 6))
    plot_solutions(S0, T, dt_fix, St_true, St_em, L)

    colors = ["g", "c", "m", "y", "k"]
    for i, N in enumerate(n_values[:-2]):
        dt = T / N
        W_coarse = W_fix[:: N_fix // N]  # Subsample Brownian motion
        dW_coarse = np.diff(W_coarse)
        St_em, L = euler_maruyama(S0, r, sigma, N, dW_coarse, dt)
        color = colors[i % len(colors)]
        plot_solutions(S0, T, dt, [], St_em, L, color=color)
    plt.title(
        "Comparison of Euler-Maruyama Solutions with Different Time Steps", fontsize=14
    )
    plt.savefig("em_comparison.png")


if __name__ == "__main__":
    main()
