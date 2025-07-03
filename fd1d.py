import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from numpy import polyfit
import pandas as pd



class DiscretizationConvection(Enum):
    UPWIND = 1
    CENTRAL_DIFFERENCE = 2
    THIRD_ORDER_UPWIND = 3
    FOURTH_ORDER_CENTRAL_DIFF = 4

class DiscretizationDiffusion(Enum):
    CENTRAL_DIFFERENCE = 1
    FOURTH_ORDER_CENTRAL_DIFF = 4



def analytical_solution(x, phi_0, phi_l, Pe, L):
    return phi_0 + ((np.exp(Pe * x / L) - 1) / (np.exp(Pe) - 1)) * (phi_l - phi_0)

def tdma(A, d):
    
    rows, _ = A.shape

    A[0,1] = A[0,1] / A[0,0]
    d[0] = d[0]/A[0,0]
    x = np.zeros((rows, 1))
    for i in range(1,rows - 1) :
        den  = A[i,i]  - A[i, i-1]*A[i-1, i]
        d_num = d[i] - A[i, i-1]*d[i -1]
        A[i, i + 1] = A[i, i + 1] / den
        d[i] = d_num/den


    d[rows -1] = (d[rows -1] - A[rows - 1, rows - 2]*d[rows -2]) / (A[rows -1 , rows -1] - A[rows -1, rows -2]*A[rows -2, rows -1])
    x[rows -1] = d[rows -1]

    for i in reversed(range(1, rows-1)):
        x[i] = d[i] - A[i , i + 1] * x [i + 1]

    return x

def solve_convection_diffusion(
                x_vals,
                phi_0,
                phi_l,
                density,
                vel,
                dif,
                discr_convection,
                diff_discr) :
    
    n = x_vals.shape[0]
    phi = np.zeros((n,1))
    n_iterations = 1
    iteration = 1
    err = 10000
    tol = 1e-10
    hasGhostPoints = diff_discr == DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF or discr_convection == DiscretizationConvection.THIRD_ORDER_UPWIND or discr_convection == DiscretizationConvection.FOURTH_ORDER_CENTRAL_DIFF
    if (hasGhostPoints):
        phi = np.zeros((n + 2,1))
        n_iterations = 10000
        

    phi_exact = analytical_solution(x_vals, phi_0=phi_0, phi_l=phi_l, Pe=Pe, L=L)

    n_unknowns = n - 2

    dx = x_vals[1] - x_vals[0]
    while iteration <= n_iterations and err > tol:
        A = np.zeros((n_unknowns, n_unknowns))  # system matrix
        b = np.zeros(n_unknowns)                # right-hand side    
        for i in range(1, n-1):    # i = 1 to n-2 #adjust this loop
            j = i - 1              # index in matrix

            dx_w = x_vals[i] - x_vals[i-1]
            dx_e = x_vals[i+1] - x_vals[i]
            dx_total = x_vals[i+1] - x_vals[i-1]

            # convection (1st-order upwind)
            if (discr_convection == DiscretizationConvection.UPWIND or discr_convection==DiscretizationConvection.THIRD_ORDER_UPWIND):
                a_ec =  min(density * vel, 0) / dx_e
                a_wc =  -max(density * vel, 0) / dx_w
                a_pc = -(a_ec + a_wc)
            elif (discr_convection == DiscretizationConvection.CENTRAL_DIFFERENCE or discr_convection == DiscretizationConvection.FOURTH_ORDER_CENTRAL_DIFF):
                a_wc = -(density * vel) /dx_total
                a_ec = (density * vel) / dx_total
                a_pc = 0.0


            # diffusion (central difference)
            a_ed = - (2 * dif) / ((dx_total) * (dx_e))
            a_wd = - (2 * dif) / ((dx_total) * (dx_w)) 
            a_pd = - (a_ed + a_wd)

            a_w = (a_wc + a_wd)
            a_e = (a_ec + a_ed)
            a_p = (a_pc + a_pd)

            A[j, j] = a_p
            if j > 0:
                A[j, j-1] = a_w
            if j < n_unknowns - 1:
                A[j, j+1] = a_e
           
            
            if i == 1:
                b[j] -= a_w * phi_0
            if i == n - 2:
                b[j] -= a_e * phi_l

            if (discr_convection == DiscretizationConvection.THIRD_ORDER_UPWIND and iteration > 1):
                ghost_idx = j + 2
                ud3 = (2 * phi[ghost_idx+1] + 3 * phi[ghost_idx] - 6 * phi[ghost_idx-1] + phi[ghost_idx-2]) / (6 * dx)
                ud1 = (phi[ghost_idx] - phi[ghost_idx-1]) / dx
                b[j] += -density * vel * (ud3 - ud1).item()

            if (discr_convection == DiscretizationConvection.FOURTH_ORDER_CENTRAL_DIFF and iteration > 1):
                ghost_idx = j + 2
                cd4 = (-phi[ghost_idx+2] + 8*phi[ghost_idx+1] - 8*phi[ghost_idx-1] + phi[ghost_idx-2]) / (12 * dx)
                cd2 = (phi[ghost_idx+1] - phi[ghost_idx-1]) / (2 * dx)
                b[j] += -density * vel * (cd4 - cd2).item()  


            if (diff_discr == DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF and iteration > 1):
                ghost_idx =  j + 2
                cd2 = (phi[ghost_idx+1] - 2 * phi[ghost_idx] + phi[ghost_idx - 1])/ (dx*dx)
                cd4 = (-phi[ghost_idx+2] + 16* phi[ghost_idx+1] - 30 * phi[ghost_idx] + 16 *phi[ghost_idx - 1] - phi[ghost_idx-2])/(12*dx*dx)
                b[j] += dif * (cd4 - cd2).item()
        
        x = tdma(A,b)
        iteration += 1
        if (hasGhostPoints):
            phi_old = phi.copy()
            phi[1] = phi_0
            phi[2:-2] = x
            phi[-2] = phi_l
            phi[0] = 5 * phi[1] - 10 * phi[2]  + 10 * phi[3] - 5 * phi[4] + phi[5]
            phi[n + 1] = phi[n - 4] - 5 * phi[n - 3] + 10 * phi[n - 2] - 10 * phi[n - 1] + 5 * phi[n]
            err = np.linalg.norm(phi - phi_old, ord = 1)
        else:
            phi[0] = phi_0
            phi[-1] = phi_l
            phi[1:-1] = x
        

    if (hasGhostPoints):
        tot_err  = np.linalg.norm(phi[1:-1].flatten() - phi_exact, ord = 1)/np.linalg.norm(phi_exact)
    else:
        tot_err  = np.linalg.norm(phi.flatten() - phi_exact, ord = 1)/np.linalg.norm(phi_exact) 
    
    return phi 
    
if __name__ == "__main__":
    density = 1.0
    vel = 1
    dif = 0.02
    phi_0 = 0.
    phi_l = 1
    x_min = 0.
    x_max = 1
    L = x_max - x_min
    Pe = (density * vel * L)/dif
    grid_sizes = [41, 81, 161, 321, 641, 1281, 1280*2 + 1, 1280 *4 +1]
    schemes = [
        ("UD1/CD2", DiscretizationConvection.UPWIND, DiscretizationDiffusion.CENTRAL_DIFFERENCE),
        ("CD2/CD2", DiscretizationConvection.CENTRAL_DIFFERENCE, DiscretizationDiffusion.CENTRAL_DIFFERENCE),
        ("UD3/CD2", DiscretizationConvection.THIRD_ORDER_UPWIND, DiscretizationDiffusion.CENTRAL_DIFFERENCE),
        ("UD3/CD4", DiscretizationConvection.THIRD_ORDER_UPWIND, DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF),
        ("CD4/CD2", DiscretizationConvection.FOURTH_ORDER_CENTRAL_DIFF, DiscretizationDiffusion.CENTRAL_DIFFERENCE),
        ("CD4/CD4", DiscretizationConvection.FOURTH_ORDER_CENTRAL_DIFF, DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF),
    ] 
    errors = {name: [] for name, _, _ in schemes}

    

    for N in grid_sizes:
       x_vals = np.linspace(x_min, x_max, N)
       phi_exact = analytical_solution(x_vals, phi_0, phi_l, Pe, L)
       for name, conv_scheme, diff_scheme in schemes:
           phi_numeric = solve_convection_diffusion(
               x_vals=x_vals,
               phi_0=phi_0,
               phi_l=phi_l,
               density=density,
               vel=vel,
               dif=dif,
               discr_convection=conv_scheme,
               diff_discr=diff_scheme
           )
           if phi_numeric.shape[0] != phi_exact.shape[0]:
               phi_numeric = phi_numeric[1:-1]  # Remove ghost nodes if needed
           err = np.linalg.norm(phi_numeric.flatten() - phi_exact, ord=1) / np.linalg.norm(phi_exact, ord=1)
           errors[name].append(err)

    #save to csv
    convergence_data = {
    'Grid Size': grid_sizes
    }

    for name in errors:
        convergence_data[f'Error ({name})'] = errors[name]
    
    df = pd.DataFrame(convergence_data)
    df.to_csv("convergence_table.csv", index=False)

    # Plotting
    plt.figure()
    for name in errors:
         # Convert to logs for least squares
        num_points_for_fit = 4 
        h_vals = 1 / (np.array(grid_sizes[-num_points_for_fit:], dtype=float) -1)
        log_h = np.log(h_vals)
        log_err = np.log(errors[name][-num_points_for_fit:])

        # Least squares fit: log(err) ≈ slope * log(h) + intercept
        slope, _ = polyfit(log_h, log_err, 1)
        slope = abs(slope)

        # Plot error vs grid size
        plt.plot(grid_sizes, errors[name], label=f"{name} (slope ≈ {slope:.2f})", marker='o')

    

    plt.xlabel("Number of Grid Points (N)")
    plt.ylabel("L1 Error Norm")
    plt.title("Convergence Study of Convection-Diffusion Schemes")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
