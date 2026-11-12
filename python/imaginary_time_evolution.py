import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm  # For progress bars
from joblib import Parallel, delayed

# ------------------- CONFIGURATION FLAGS -------------------

do_plot = False             # Plots are made at the end, after all V0 values are done
overwrite_existing = False  # Set to True to overwrite existing wavefunctions
n_jobs = -1                 # Number of parallel workers (-1 = all CPU cores)

# ------------------- CONSTANTS -------------------

grid_size = 20       # Fixed grid_size x grid_size spatial grid
dtau = 2e-5            # Imaginary time step (stable for dtau < dx^2/2)
hbar = 1.0             # Planck constant (reduced)
m = 1.0                # Particle mass
k = 2 * np.pi          # Cavity wavevector
num_steps = 10**6      # Total imaginary time tau = num_steps * dtau

# ------------------- V0 VALUES (Kac) -------------------
# The two-body problem is solved at the bare coupling V0 = V0_kac / N, so that
# a run with N particles corresponds to the Kac-scaled coupling V0_kac.

N_list      = [10, 15, 20, 25, 30, 35, 40, 45, 50]
V0_kac_list = np.arange(0.0, -251.0, -10.0)

V0_array = sorted({round(V0_kac / N, 3) for N in N_list for V0_kac in V0_kac_list})

# ------------------- LAPLACIAN OPERATORS -------------------

def laplacian(psi, dx):
    return (np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
            np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) -
            4 * psi) / dx**2

# ------------------- FIXED GRID SETUP -------------------

L = 1.0                                # Unit cell: run_vmc.jl interpolates on [-1/2, 1/2)
dx = L / grid_size                    # Spatial resolution
x = np.linspace(-L/2, L/2, grid_size, endpoint=False)
x1, x2 = np.meshgrid(x, x, indexing='ij')  # 2D coordinate grid


def wavefunction_path(V0):
    save_path = f'numpy_arrays_VMC/wavefunction/V0={V0:.3f}'
    return save_path, f"{save_path}/wavefunction_{V0:.3f}_{L:.1f}.npy"


# ------------------- IMAGINARY TIME EVOLUTION FOR ONE V₀ -------------------

def solve_V0(V0):
    """Imaginary-time evolution of the two-body problem at bare coupling V0; saves psi."""
    V0 = float(V0)
    v0_start = time.time()
    save_path, wavefunction_file = wavefunction_path(V0)

    V = V0 * np.cos(k * x1) * np.cos(k * x2)

    psi = np.ones([grid_size, grid_size])
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

    for step in range(num_steps):
        T_psi = - (hbar**2 / (2 * m)) * laplacian(psi, dx)
        H_psi = T_psi + V * psi
        psi -= (dtau / hbar) * H_psi
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx**2)

    os.makedirs(save_path, exist_ok=True)
    np.save(wavefunction_file, psi)

    T_psi = - (hbar**2 / (2 * m)) * laplacian(psi, dx)
    H_psi = T_psi + V * psi
    E_total = np.sum(np.conj(psi) * H_psi).real * dx**2
    E_int   = np.sum(np.conj(psi) * V * psi).real * dx**2

    v0_elapsed = time.time() - v0_start
    return f"V0 = {V0:.6f} | E_total = {E_total:.6f}, E_int = {E_int:.6f} | Time: {v0_elapsed:.2f} s"


# ------------------- MAIN: RUN ALL V₀ IN PARALLEL -------------------

if __name__ == "__main__":
    start_time = time.time()

    todo = []
    for V0 in V0_array:
        _, wavefunction_file = wavefunction_path(V0)
        if os.path.exists(wavefunction_file) and not overwrite_existing:
            print(f"V0 = {V0:.6f} (already exists, skipping)")
        else:
            todo.append(V0)

    print(f"\n{len(todo)} of {len(V0_array)} V0 values to compute")

    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(solve_V0)(V0) for V0 in todo
    )
    for line in tqdm(results, total=len(todo), desc="V0 values"):
        print(f"\n{line}")

    # ------------------- PLOTS (optional, after all runs) -------------------

    if do_plot:
        for V0 in todo:
            _, wavefunction_file = wavefunction_path(V0)
            psi = np.load(wavefunction_file)
            V = V0 * np.cos(k * x1) * np.cos(k * x2)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            im0 = axes[0].imshow(V, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='plasma')
            fig.colorbar(im0, ax=axes[0], label=r'$V(x_1, x_2)$')
            axes[0].set_title(f'Interaction Energy, $V_0$={V0:.3f}')
            im1 = axes[1].imshow(np.abs(psi)**2, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='plasma')
            fig.colorbar(im1, ax=axes[1], label=r'$|\psi(x_1, x_2)|^2$')
            axes[1].set_title(f'Final Wavefunction, $V_0$={V0:.3f}')
            for ax in axes:
                ax.set_xlabel(r'$x_1$')
                ax.set_ylabel(r'$x_2$')
            plt.tight_layout()
            plt.show()

    # ------------------- FINAL TIMING -------------------

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")