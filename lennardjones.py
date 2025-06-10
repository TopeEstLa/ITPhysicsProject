# Créé par zlach, le 09/06/2025 en Python 3.7

from numpy import linspace, zeros, diag, arange
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import numpy as np

# Paramètres physiques sans dimension
N = 2000                      # Nombre de points
x_min, x_max = 0.7, 4.0       # Domaine spatial (en unités de sigma)
x = linspace(x_min, x_max, N)
dx = x[1] - x[0]

# Potentiel de Lennard-Jones sans dimension
def U(x):
    return 4 * (1 / x**12 - 1 / x**6)

V = U(x)                      # Tableau du potentiel

# Construction de la matrice Hamiltonienne H = T + V
# Matrice cinétique (differences finies centrées)
T = diag([1]* (N-1), -1) + diag([-2]*N, 0) + diag([1]* (N-1), 1)
T = - T / dx**2               # opérateur -d²/dx²

# Matrice potentielle (diagonale)
V_diag = diag(V)

# Hamiltonien total
H = T + V_diag

# Résolution de l’équation de Schrödinger : valeurs et vecteurs propres
valeurs_propres, vecteurs_propres = eigh(H)

# Affichage des 6 premiers états liés
plt.figure(figsize=(10, 6))
plt.plot(x, V, label='Potentiel V(x)', color='black')

for n in range(10):
    E = valeurs_propres[n]
    psi = vecteurs_propres[:, n]
    psi = psi / max(abs(psi))      # normalisation visuelle
    plt.plot(x, psi + E, label=f'État {n}, E = {E:.3f}')

plt.xlabel("x = r / σ")
plt.ylabel("Énergie / Fonction d’onde")
plt.title("États liés dans le potentiel de Lennard-Jones")
plt.legend()
plt.grid()
plt.show()






## TRANSMISSIO?N



# Paramètres
N = 2000
x_min, x_max = -10, 10
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]



# Potentiel de Lennard-Jones tronqué
def U(x):
    r = np.abs(x) + 0.7  # éviter x=0
    return 4 * (1/r**12 - 1/r**6)

V = U(x)

# Résolution de l’équation de Schrödinger stationnaire pour une onde incidente
def transmission(E):
    k2 = 2 * (E - V)  # ici ħ²/2m = 1
    diag_main = -2 / dx**2 + k2
    diag_off = np.ones(N - 1) / dx**2
    ab = np.zeros((3, N))
    ab[0, 1:] = diag_off       # bande supérieure
    ab[1, :] = diag_main       # diagonale principale
    ab[2, :-1] = diag_off      # bande inférieure

    # Onde incidente de gauche : exp(ikx)
    k = np.sqrt(2 * E)
    psi0 = np.exp(1j * k * x)
    psi0[0] = np.exp(1j * k * x[0])
    psi0[-1] = np.exp(1j * k * x[-1])

    psi = solve_banded((1, 1), ab, psi0)

    # Transmission estimée à droite (rapport d'amplitudes)
    T = np.abs(psi[-100])**2 / np.abs(psi0[-100])**2
    return T

# Balayage en énergie
energies = np.linspace(0.01, 1.5, 300)
transmissions = [transmission(E) for E in energies]

# Tracé
plt.figure(figsize=(8, 5))
plt.plot(energies, transmissions)
plt.title("Effet Ramsauer – Coefficient de transmission")
plt.xlabel("Énergie E")
plt.ylabel("Transmission T(E)")
plt.grid()
plt.show()
