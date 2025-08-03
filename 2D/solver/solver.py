import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab

def idx(i, j, Ny):
    return i * Ny + j

def convection_diffusion_solver(Nx, Ny, D, u, v, boundary_types, boundary_funcs, source_term):
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
    N = Nx * Ny
    A = lil_matrix((N, N))
    b = np.zeros(N)

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j, Ny)
            x, y = i * dx, j * dy  # physical coordinates

            # --- Left boundary ---
            if i == 0:
                if boundary_types["left"] == "dirichlet":
                    A[p, p] = 1.0
                    b[p] = boundary_funcs["left"](x, y)
                elif boundary_types["left"] == "neumann":
                    # 2nd order: (-3φ0 + 4φ1 - φ2) / (2dx) = g
                    A[p, p] = -3.0 / (2.0 * dx)
                    A[p, idx(i+1, j, Ny)] = 4.0 / (2.0 * dx)
                    A[p, idx(i+2, j, Ny)] = -1.0 / (2.0 * dx)
                    b[p] = boundary_funcs["left"](x, y)
                continue

            # --- Right boundary ---
            if i == Nx - 1:
                if boundary_types["right"] == "dirichlet":
                    A[p, p] = 1.0
                    b[p] = boundary_funcs["right"](x, y)
                elif boundary_types["right"] == "neumann":
                    # 2nd order: (3φN - 4φN-1 + φN-2) / (2dx) = g
                    A[p, p] = 3.0 / (2.0 * dx)
                    A[p, idx(i-1, j, Ny)] = -4.0 / (2.0 * dx)
                    A[p, idx(i-2, j, Ny)] = 1.0 / (2.0 * dx)
                    b[p] = boundary_funcs["right"](x, y)
                continue

            # --- Bottom boundary ---
            if j == 0:
                if boundary_types["bottom"] == "dirichlet":
                    A[p, p] = 1.0
                    b[p] = boundary_funcs["bottom"](x, y)
                elif boundary_types["bottom"] == "neumann":
                    # 2nd order: (-3φ0 + 4φ1 - φ2) / (2dy) = g
                    A[p, p] = -3.0 / (2.0 * dy)
                    A[p, idx(i, j+1, Ny)] = 4.0 / (2.0 * dy)
                    A[p, idx(i, j+2, Ny)] = -1.0 / (2.0 * dy)
                    b[p] = boundary_funcs["bottom"](x, y)
                continue

            # --- Top boundary ---
            if j == Ny - 1:
                if boundary_types["top"] == "dirichlet":
                    A[p, p] = 1.0
                    b[p] = boundary_funcs["top"](x, y)
                elif boundary_types["top"] == "neumann":
                    # 2nd order: (3φN - 4φN-1 + φN-2) / (2dy) = g
                    A[p, p] = 3.0 / (2.0 * dy)
                    A[p, idx(i, j-1, Ny)] = -4.0 / (2.0 * dy)
                    A[p, idx(i, j-2, Ny)] = 1.0 / (2.0 * dy)
                    b[p] = boundary_funcs["top"](x, y)
                continue

            # --- Interior nodes ---
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

    # --- Solve linear system ---
    phi_vec, info = bicgstab(A.tocsr(), b, rtol=1e-8)
    if info != 0:
        raise RuntimeError(f"Solver did not converge. Info: {info}")

    return phi_vec.reshape((Nx, Ny))
