import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab

# --- Helper: map (i,j) to linear index ---
def idx(i, j, Ny):
    return i * Ny + j


def convection_diffusion_solver(Nx, Ny, D, u ,v,dirichlet,source_term):    
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
    N = Nx * Ny
    A = lil_matrix((N, N))
    b = np.zeros(N)

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j, Ny)

            # Boundary condition
            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                A[p, p] = 1.0
                b[p] = dirichlet[i, j]
                continue

            # Coefficients: upwind convection + CD2 diffusion
            aw = D / dx**2 + (u / dx if u > 0 else 0)
            ae = D / dx**2 + (0 if u > 0 else -u / dx)
            as_ = D / dy**2 + (v / dy if v > 0 else 0)
            an = D / dy**2 + (0 if v > 0 else -v / dy)
            ap = aw + ae + as_ + an

            A[p, idx(i, j, Ny)]     = ap
            A[p, idx(i-1, j, Ny)]   = -aw
            A[p, idx(i+1, j, Ny)]   = -ae
            A[p, idx(i, j-1, Ny)]   = -as_
            A[p, idx(i, j+1, Ny)]   = -an
            b[p] = source_term[i, j]

        # --- Solve linear system using BiCGSTAB ---
    phi_vec, info = bicgstab(A.tocsr(), b, rtol=1e-8)

        # --- Reshape to 2D solution ---
    phi = phi_vec.reshape((Nx, Ny))
    phi, info = bicgstab(A.tocsr(), b, rtol=1e-10)
    if info != 0:
        raise RuntimeError(f"Solver did not converge. Info: {info}")

    return phi.reshape((Ny, Nx))
