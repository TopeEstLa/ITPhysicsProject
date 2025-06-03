import math
import os
import time

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

matplotlib.use('TkAgg')

def init():
    line.set_data([], [])
    return line,


def animate(j):
    line.set_data(o, final_densite[j, :])
    return line,


dt = 1E-7
dx = 0.001
nt = 50000
nd = int(nt / 1000) + 1
xc = 0.6 # Position du paquet d'onde
sigma = 0.05
v0 = -4000  # Potentiel
e = 5
pi = math.pi

nx = int(1 / dx) * 2
n_frame = nd
s = dt / (dx ** 2)
A = 1 / (math.sqrt(sigma * math.sqrt(pi)))
E = e * v0
k = math.sqrt(2 * abs(E))  # on approxime que h(barre) & m = 1

o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= 0.8) & (o <= 0.9)] = v0  # Potentiel

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
densite = np.zeros((nt, nx))
densite[0, :] = np.abs(cpt[:]) ** 2
final_densite = np.zeros((n_frame, nx))
re = np.real(cpt[:])
b = np.zeros(nx)
im = np.imag(cpt[:])

# Solveur d'EDP
it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i, 1:-1] = re[1:-1] * re[1:-1] + im[1:-1] * b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

for i in range(1, nt):
    if (i - 1) % 1000 == 0:
        it += 1
        final_densite[it][:] = densite[i][:]

plot_title = "Marche Ascendante avec E/Vo=" + str(e)

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_ylim(-6, 12)
ax.set_xlim(0, 2)
ax.plot(o, V, label="Potentiel")
ax.set_title(plot_title)
ax.set_xlabel("x")
ax.set_ylabel("Densité de probabilité de présence")
ax.legend()

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=False, interval=100, repeat=False)
file_name = 'paquet_onde_e='+str(e)+'.gif'
ani.save(file_name, fps=10)
print(f"Animation exportée dans data")

plt.show()


# Etats stationnaires les 3 premier états stationnaires sont ~= bon comparé au calcul analytique après bizarre
n_states = 5

x = np.linspace(0, (nx - 1) * dx, nx)

diag = np.full(nx, -2.0)
offdiag = np.full(nx - 1, 1.0)
T = (-1 / dx ** 2) * (np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1))
H = T + np.diag(V) # hamiltonien

energies, states = eigh(H, subset_by_index=(0, n_states - 1))


first_n_found = False
plt.figure(figsize=(10, 6))
for i in range(n_states):
    if energies[i] > 0:
        if not first_n_found:
            first_n_found = True
            print("Première énergie positive trouvée pour l'état n=", i + 1, "E =", energies[i])
#        continue

    psi = states[:, i]
    psi = psi / np.sqrt(np.sum(psi ** 2) * dx)

    plt.plot(x, psi ** 2 + energies[i], label=f"État n={i+1} (E = {energies[i]:.2f})")

plt.plot(x, V, 'k--', label='Potentiel V(x)')
plt.title("États stationnaires")
plt.xlabel("x")
plt.ylabel("Énergie / Densité de probabilité")
plt.legend()
plt.grid()
file_name = 'etats_stationnaires_e='+str(e)+'.png'
plt.savefig(file_name)
plt.show()
print(f"Graphique des états stationnaires exporté dans '{file_name}'")