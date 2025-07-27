import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots


# --- Grid setup ---
Lx, Ly = 1.0, 1.0
Nx, Ny = 150, 150
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# --- Physical parameters ---
u, v = 1.0, 1.0      # convection velocities
D = 0.1              # diffusion coefficient

# --- Analytical solution and source term ---
phi_exact = np.cos(np.pi * X) * np.cos(np.pi * Y)
source = (
    -np.pi * u * np.sin(np.pi * X) * np.cos(np.pi * Y)
    -np.pi * v * np.cos(np.pi * X) * np.sin(np.pi * Y)
    + 2 * D * np.pi**2 * np.cos(np.pi * X) * np.cos(np.pi * Y)
)

# --- Helper: map (i,j) to linear index ---
def idx(i, j):
    return i * Ny + j

# --- Assemble sparse matrix and RHS vector ---
N = Nx * Ny
A = lil_matrix((N, N))
b = np.zeros(N)

for i in range(Nx):
    for j in range(Ny):
        p = idx(i, j)

        # Boundary condition
        if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
            A[p, p] = 1.0
            b[p] = phi_exact[i, j]
            continue

        # Coefficients: upwind convection + CD2 diffusion
        aw = D / dx**2 + (u / dx if u > 0 else 0)
        ae = D / dx**2 + (0 if u > 0 else -u / dx)
        as_ = D / dy**2 + (v / dy if v > 0 else 0)
        an = D / dy**2 + (0 if v > 0 else -v / dy)
        ap = aw + ae + as_ + an

        A[p, idx(i, j)]     = ap
        A[p, idx(i-1, j)]   = -aw
        A[p, idx(i+1, j)]   = -ae
        A[p, idx(i, j-1)]   = -as_
        A[p, idx(i, j+1)]   = -an
        b[p] = source[i, j]

# --- Solve linear system using BiCGSTAB ---
phi_vec, info = bicgstab(A.tocsr(), b, rtol=1e-8)

# --- Reshape to 2D solution ---
phi = phi_vec.reshape((Nx, Ny))

# --- Compute L1 error norm ---
l1_error = np.sum(np.abs(phi - phi_exact)) / np.sum(np.abs(phi_exact))

# --- Plot numerical solution ---
# --- 3D Surface Plot with Error Annotation ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, phi, cmap='viridis', edgecolor='k', linewidth=0.3)

# Title with error value
ax.set_title(f"BiCGSTAB Numerical Solution φ(x, y)\nL1 Error = {l1_error:.4e}", pad=20)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("φ")

# Colorbar
fig.colorbar(surf, shrink=0.6, label=r"$\phi$")
plt.tight_layout()
plt.show()