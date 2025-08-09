import numpy as np
import matplotlib.pyplot as plt
import os
from solver.solver import convection_diffusion_solver  # Updated GMRES+ILU version

def compute_and_plot_wall_flux(phi, D, dx, dy):
    """
    Compute and plot wall flux along the left (x=0) boundary.

    Parameters:
    - phi: 2D array of scalar field values, shape (nx, ny)
    - D: diffusion coefficient
    - dx: grid spacing in x-direction
    - dy: grid spacing in y-direction
    """
    nx, ny = phi.shape
    y = np.linspace(0, 1, ny)

    # Use 2nd-order accurate one-sided difference:
    # dphi/dx ≈ (-3φ0 + 4φ1 - φ2) / (2Δx)
    phi0 = phi[0, :]
    phi1 = phi[1, :]
    #phi2 = phi[2, :]

    #dphi_dx = (-3 * phi0 + 4 * phi1 - phi2) / (2 * dx)
    dphi_dx = (phi1 - phi0) / dx
    wall_flux = D * dphi_dx

    # --- Plot ---
    plt.figure(figsize=(6, 5))
    plt.plot(wall_flux, y, color='black', linewidth=2)
    plt.xlabel("Wall flux [ -$\Gamma$ ∂φ/∂x ]")
    plt.ylabel("y")
    plt.title("Left Wall Diffusive Flux Profile")
    plt.grid(True)
    plt.gca().invert_yaxis()  # So y=0 is at bottom
    plt.tight_layout()
    os.makedirs("2D/results", exist_ok=True)
    plt.savefig("2D/results/stagnation_point_wall_flux.png")

    plt.show()

    return wall_flux

def plot_colored_contour_lines(X, Y, phi, num_levels=20, filename=None, cmap='viridis'):
    """
    Plot colored contour lines of phi(x,y) with color indicating magnitude.
    
    Args:
        X, Y: 2D meshgrid arrays (X.shape == Y.shape == phi.shape)
        phi: 2D scalar field (Nx × Ny)
        num_levels: number of contour levels
        filename: optional path to save the plot
        cmap: matplotlib colormap name
    """
    # Transpose phi for correct orientation (if needed)
    phi_plot = phi.T

    # Compute contour levels
    vmin, vmax = np.nanmin(phi_plot), np.nanmax(phi_plot)
    levels = np.linspace(vmin, vmax, num_levels)

    # Create figure
    plt.figure(figsize=(6, 5))
    
    # Colored contour lines
    contour = plt.contour(X, Y, phi_plot, levels=levels, cmap=cmap)
    
    # Colorbar linked to contour values
    plt.colorbar(contour, label=r'$\phi$')

    # Axis settings
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Temperature Isotherms (Colored Contour Lines)")
    plt.axis("equal")
    plt.grid(True)
    os.makedirs("2D/results", exist_ok=True)
    plt.savefig("2D/results/stagnation_pint_contour.png")

    plt.show()


def test_stagnation_point():
    # --- Grid ---
    nx, ny = 40,40
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Physical parameters ---
    rho = 100
    Gamma = 0.1
    D = Gamma / rho

    # --- Velocity field ---
    u_field = lambda x, _ : x           # u = x
    v_field = lambda _ , y : -y         # v = -y

    # --- No source term ---
    source_term = np.zeros_like(X)

    # --- Boundary conditions ---
    boundary_types = {
        "left": "dirichlet",   # Inlet
        "right": "neumann",    # Outlet
        "bottom": "neumann",   # Symmetry
        "top": "dirichlet"     # Wall
    }

    boundary_funcs = {
        "left": lambda x, y: 1-y,  # isothermal at inlet
        "right": lambda x, y: 0,   # zero gradient at outlet
        "bottom": lambda x, y: 0,  # symmetry
        "top": lambda x, y: 0      # wall at constant temperature
    }

    # --- Run solver ---
    phi_num = convection_diffusion_solver(
        nx, ny,
        D, u_field, v_field,
        boundary_types=boundary_types,
        boundary_funcs=boundary_funcs,
        source_term=source_term
    )

    print("Any NaNs in phi_num?", np.isnan(phi_num).any())
    print("Any Infs in phi_num?", np.isinf(phi_num).any())

    # --- Plot ---
    plot_colored_contour_lines(X, Y, phi_num, num_levels=30)

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    wall_flux = compute_and_plot_wall_flux(phi_num, Gamma, dx, dy)

    
if __name__ == "__main__":
    test_stagnation_point()
