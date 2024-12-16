import numpy as np
import matplotlib.pyplot as plt

# Parameters for both methods
np.random.seed(100)
T = 1
M = 1000  # Number of paths for Milstein and Euler-Maruyama
N = 2**11
lambda_, mu, Xzero = 2, 1, 1  # Parameters for Euler-Maruyama
r, K, beta = 2, 1, 0.25  # Parameters for Milstein

# Shared Brownian increments
dt = T / N
dW = np.sqrt(dt) * np.random.randn(M, N)

# Milstein and Euler-Maruyama Error Computation
R = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
Xerr_mil = np.zeros((M, len(R) - 1))
Xerr_em = np.zeros((M, len(R) - 1))

for p, R_factor in enumerate(R):
    Dt = R_factor * dt
    L = N // R_factor
    Xtemp_mil = Xzero * np.ones(M)
    Xtemp_em = Xzero * np.ones(M)

    for j in range(L):
        Winc = np.sum(dW[:, R_factor * j : R_factor * (j + 1)], axis=1)
        
        # Milstein update
        if p > 0:
            Xtemp_mil += (
                Dt * r * Xtemp_mil * (K - Xtemp_mil)
                + beta * Xtemp_mil * Winc
                + 0.5 * beta**2 * Xtemp_mil * (Winc**2 - Dt)
            )
        
        # Euler-Maruyama update
        Xtemp_em += Dt * lambda_ * Xtemp_em + mu * Xtemp_em * Winc

    if p > 0:
        Xref_mil = Xtemp_mil if p == 1 else Xref_mil
        Xerr_mil[:, p - 1] = np.abs(Xtemp_mil - Xref_mil)
        
        Xtrue = Xzero * np.exp((lambda_ - 0.5 * mu**2) * T + mu * np.sum(dW, axis=1))
        Xerr_em[:, p - 1] = np.abs(Xtemp_em - Xtrue)

mean_Xerr_mil = np.mean(Xerr_mil, axis=0)
mean_Xerr_em = np.mean(Xerr_em, axis=0)
Dtvals = dt * R[1:]

# Plotting the Errors for Comparison
plt.figure(figsize=(10, 6))
plt.loglog(Dtvals, mean_Xerr_mil, "b*-", label="Milstein Error")
plt.loglog(Dtvals, mean_Xerr_em, "g*-", label="Euler-Maruyama Error")
plt.xlabel("$\\Delta t$", fontsize=12)
plt.ylabel("Sample average of $|X(T) - X_L|$", fontsize=12)
plt.title("Comparison of Strong Convergence: Milstein vs Euler-Maruyama", fontsize=14)
plt.legend()
plt.grid()
plt.show()
