import numpy as np
from solver.solver import convection_diffusion_solver

def analytical_solution(X, Y):
    return np.cos(np.pi * X) * np.cos(np.pi * Y)

def source_from_solution(nx, ny, sol_func):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    phi = sol_func(X, Y)

    d2phi_dx2 = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / dx**2
    d2phi_dy2 = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / dy**2
    return -(d2phi_dx2 + d2phi_dy2)

def test_dirichlet():
    nx, ny = 100, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- Physical parameters ---
    u = lambda x,y: 1.0
    v = lambda x,y: 1.0
    D = 0.1              # diffusion coefficient

# --- Analytical solution and source term ---
    phi_exact = np.cos(np.pi * X) * np.cos(np.pi * Y)
    source_term = (
    -np.pi * u(X,Y) * np.sin(np.pi * X) * np.cos(np.pi * Y)
    -np.pi * v(X,Y) * np.cos(np.pi * X) * np.sin(np.pi * Y)
    + 2 * D * np.pi**2 * np.cos(np.pi * X) * np.cos(np.pi * Y)
    )

    boundary_types = {
    "left": "dirichlet",
    "right": "dirichlet",
    "bottom": "dirichlet",
    "top": "dirichlet"
    }

    boundary_funcs = {
    "left":  lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y),
    "right": lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y),
    "bottom":lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y),
    "top":   lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y)
    }
    
    phi_num = convection_diffusion_solver(nx, ny,D, u ,v, boundary_types= boundary_types, boundary_funcs= boundary_funcs, source_term=source_term)

    error = np.linalg.norm(phi_num - phi_exact) / np.linalg.norm(phi_exact)
    assert error < 1e-2

    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

    # --- Plot numerical solution ---
    # --- 3D Surface Plot with Error Annotation ---
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    
    # Calculate L1 error
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, phi_exact, cmap='viridis', edgecolor='k', linewidth=0.3)

    wire = ax.plot_wireframe(X, Y, phi_num, color='red', linewidth=0.7, label='Numerical')

    # Title with error value
    ax.set_title(f"BiCGSTAB Numerical Solution φ(x, y)\nL1 Error = {error:.4e}", pad=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("φ")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Numerical (wireframe)'),
        Line2D([0], [0], color='blue', lw=2, label='Analytical (surface)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    # Colorbar
    plt.tight_layout()
    os.makedirs("2D/results", exist_ok=True)
    plt.savefig("2D/results/test_dirichlet.png")



