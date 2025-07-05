import numpy as np
import matplotlib.pyplot as plt
import os
import time
from numba import njit, prange
from tqdm import tqdm  # For progress bars

# ------------------- CONFIGURATION FLAGS -------------------

do_plot = True              # Set to False to disable plotting
overwrite_existing = True  # Set to True to overwrite existing wavefunctions

# ------------------- CONSTANTS -------------------

grid_size = 100        # Use a fixed 100x100 spatial grid
dtau = 1e-7            # Imaginary time step
hbar = 1.0             # Planck constant (reduced)
m = 1.0                # Particle mass
k = 2 * np.pi          # Cavity wavevector
num_steps = 10**6      # Number of imaginary time evolution steps

# V₀ values to simulate
V0_array = np.arange(10.0, 11.0, 1.0)

# ------------------- LAPLACIAN OPERATORS -------------------

def laplacian(psi, dx):
    return (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
            np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) -
            4 * psi) / dx**2

# ------------------- FIXED GRID SETUP -------------------

L = 3.0                                # Size of the 2D box
dx = L / grid_size                    # Spatial resolution
x = np.linspace(-L/2, L/2, grid_size, endpoint=False)
x1, x2 = np.meshgrid(x, x, indexing='ij')  # 2D coordinate grid

# ------------------- MAIN LOOP OVER V₀ -------------------

start_time = time.time()


for V0 in V0_array:
    V0 = float(V0)
    v0_start = time.time()

    save_path = f'numpy_arrays_VMC/wavefunction/V0={V0:.1f}'
    wavefunction_file = f"{save_path}/wavefunction_{V0:.1f}_{L:.1f}.npy"

    if os.path.exists(wavefunction_file) and not overwrite_existing:
        print(f"V0 = {V0:.1f} (already exists, skipping)")
        pbar.update(1)
        continue

    # ------------------- DEFINE POTENTIAL -------------------

    V = V0 * np.cos(k * x1) * np.cos(k * x2)

    if do_plot:
        plt.figure(figsize=(8, 7))
        plt.imshow(V, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='plasma')
        plt.colorbar(label=r'$V(x_1, x_2)$')
        plt.title(f'Interaction Energy, $V_0$={V0:.1f}')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.tight_layout()
        plt.show()

    # ------------------- INITIALIZE WAVEFUNCTION -------------------

    psi = np.ones([grid_size, grid_size])
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

    # ------------------- IMAGINARY TIME EVOLUTION -------------------

    for step in tqdm(range(num_steps), desc=f"Imaginary time V₀={V0:.1f}", leave=False, dynamic_ncols=True):
        T_psi = - (hbar**2 / (2 * m)) * laplacian(psi, dx)
        H_psi = T_psi + V * psi
        psi -= (dtau / hbar) * H_psi
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

    # ------------------- SAVE WAVEFUNCTION -------------------

    os.makedirs(save_path, exist_ok=True)
    np.save(wavefunction_file, psi)

    # ------------------- FINAL WAVEFUNCTION PLOT -------------------

    if do_plot:
        plt.figure(figsize=(8, 7))
        plt.imshow(np.abs(psi)**2, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='plasma')
        plt.colorbar(label=r'$|\psi(x_1, x_2)|^2$')
        plt.title(f'Final Wavefunction, $V_0$={V0:.1f}')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.tight_layout()
        plt.show()

    # ------------------- ENERGY ESTIMATION -------------------

    T_psi = - (hbar**2 / (2 * m)) * laplacian(psi, dx)
    H_psi = T_psi + V * psi
    E_total = np.sum(np.conj(psi) * H_psi).real * dx**2
    E_int   = np.sum(np.conj(psi) * V * psi).real * dx**2

    v0_elapsed = time.time() - v0_start
    print(f"V0 = {V0:.1f} | E_total = {E_total:.6f}, E_int = {E_int:.6f} | Time: {v0_elapsed:.2f} s")

# ------------------- FINAL TIMING -------------------

elapsed = time.time() - start_time
print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
