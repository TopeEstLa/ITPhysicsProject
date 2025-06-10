# Créé par zlach, le 09/06/2025 en Python 3.7

from numpy import linspace, zeros, diag, arange
from numpy.linalg import eigh
import matplotlib.pyplot as plt

# Paramètres physiques sans dimension
N = 1000                      # Nombre de points
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

# Affichage des 3 premiers états liés
plt.figure(figsize=(10, 6))
plt.plot(x, V, label='Potentiel V(x)', color='black')

for n in range(3):
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
