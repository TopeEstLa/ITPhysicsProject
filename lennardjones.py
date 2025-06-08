import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use('TkAgg')

def V_LJ(r, epsilon, sigma):
    r_safe = np.maximum(r, 1e-10)
    return 4 * epsilon * ((sigma / r_safe)**12 - (sigma / r_safe)**6)

def schrodinger_rhs(r, y):
    u, v = y
    V = V_LJ(r, epsilon, sigma)
    du = v
    dv = 2 * m / hbar**2 * (V - E) * u
    return [du, dv]

m = 1.0
hbar = 1.0
epsilon = 0.1
sigma = 1.0
E = -0.03

r_min = 0.3  # 🔥 plus loin de la divergence
r_max = 10
y0 = [0.0, 1.0]

sol = solve_ivp(schrodinger_rhs, (r_min, r_max), y0, method='RK45', max_step=0.05)

if sol.success:
    plt.plot(sol.t, sol.y[0], label='u(r)')
    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.title('État stationnaire – Potentiel Lennard-Jones')
    plt.grid()
    plt.legend()
    plt.show()
else:
    print("Erreur d'intégration :", sol.message)