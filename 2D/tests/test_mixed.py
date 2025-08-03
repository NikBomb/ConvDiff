import numpy as np
from solver.solver import convection_diffusion_solver

def analytical_solution(X, Y):
    return np.cos(np.pi * X) * np.cos(np.pi * Y)

def dphi_dx(X, Y):
    return -np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)

def dphi_dy(X, Y):
    return -np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)

def test_mixed():
    nx, ny = 100, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- Physical parameters ---
    u, v = 1.0, 1.0      # convection velocities
    D = 0.1              # diffusion coefficient

    # --- Analytical solution ---
    phi_exact = analytical_solution(X, Y)

    # --- Source term ---
    source_term = (
        -np.pi * u * np.sin(np.pi * X) * np.cos(np.pi * Y)  # convection in x
        -np.pi * v * np.cos(np.pi * X) * np.sin(np.pi * Y)  # convection in y
        + 2 * D * np.pi**2 * np.cos(np.pi * X) * np.cos(np.pi * Y)  # diffusion
    )

    # --- Mixed BCs ---
    # Left & right -> Dirichlet
    # Top & bottom -> Neumann
    boundary_types = {
        "left":   "dirichlet",
        "right":  "dirichlet",
        "bottom": "neumann",
        "top":    "neumann"
    }

    # Dirichlet BCs get φ values, Neumann BCs get ∂φ/∂n
    boundary_funcs = {
        "left":   lambda x, y: analytical_solution(x, y),
        "right":  lambda x, y: analytical_solution(x, y),
        "bottom": lambda x, y: -(-np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)),  # -∂φ/∂y
        "top":    lambda x, y: -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),     # ∂φ/∂y
    }

    # --- Solve ---
    phi_num = convection_diffusion_solver(
        nx, ny, D, u, v,
        boundary_types=boundary_types,
        boundary_funcs=boundary_funcs,
        source_term=source_term
    )

    # --- Error ---
    error = np.linalg.norm(phi_num - phi_exact) / np.linalg.norm(phi_exact)
    assert error < 2e-2, f"Error too high: {error}"

    # --- Plot ---
    from matplotlib import pyplot as plt
    import os
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, phi_exact, cmap='viridis', edgecolor='k', linewidth=0.3)
    wire = ax.plot_wireframe(X, Y, phi_num, color='red', linewidth=0.7, label='Numerical')

    ax.set_title(f"BiCGSTAB Numerical Solution φ(x, y) - Mixed BCs\nL2 Error = {error:.4e}", pad=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("φ")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Numerical (wireframe)'),
        Line2D([0], [0], color='blue', lw=2, label='Analytical (surface)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    os.makedirs("2D/results", exist_ok=True)
    plt.savefig("2D/results/test_mixed_bc.png")
