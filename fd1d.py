import numpy as np
import matplotlib.pyplot as plt
import time
from enum import Enum


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

if __name__ == "__main__":
    density = 1.0
    vel = 10
    dif = 0.2
    phi_0 = 0.
    phi_l = 1
    x_min = 0.
    x_max = 1
    n = 100
    L = x_max - x_min
    Pe = (density * vel * L)/dif 
    discr_convection = DiscretizationConvection.UPWIND
    diff_discr = DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF
    phi = np.zeros((n,1))
    n_iterations = 1
    iteration = 1
    err = 1000
    tol = 1e-10

    if (diff_discr == DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF):
        phi = np.zeros((n + 2,1))
        n_iterations = 10000
        


    x_vals = np.linspace(x_min, x_max, n)    
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
            a_wc = - (density * vel) / dx_w
            a_ec = 0.0

            # diffusion (central difference)
            dxr = 2.0 / dx_total
            a_wd = -dif * dxr / dx_w
            a_ed = -dif * dxr / dx_e

            a_w = a_wc + a_wd
            a_e = a_ec + a_ed
            a_p = -a_w - a_e

            A[j, j] = a_p
            if j > 0:
                A[j, j-1] = a_w
            if j < n_unknowns - 1:
                A[j, j+1] = a_e
           
            
            if i == 1:
                b[j] -= a_w * phi_0
            if i == n - 2:
                b[j] -= a_e * phi_l

            if (diff_discr == DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF and iteration > 1):
                ghost_idx =  j + 2
                b[j] += dif * (
                (-phi[ghost_idx + 2] + 16*phi[ghost_idx + 1] - 30*phi[ghost_idx] + 16*phi[ghost_idx - 1] - phi[ghost_idx - 2]) / (12 * dx**2)
                - (phi[ghost_idx + 1] - 2*phi[ghost_idx] + phi[ghost_idx - 1]) / (dx**2))                
        
        x = tdma(A,b)
        iteration += 1
        if (diff_discr == DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF):
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
        


    tot_err  = np.linalg.norm(phi - phi_exact, ord = 1)/np.linalg.norm(phi_exact)

    print("Total Error:", tot_err)
    # Plot
    plt.figure()
    plt.plot(x_vals, phi_exact, label=f'Analytical Solution (Pe={Pe})')
    if (diff_discr == DiscretizationDiffusion.FOURTH_ORDER_CENTRAL_DIFF):
        plt.plot(x_vals[:], phi[1:-1], 'o-', label= 'Numerical FD')
    else:
        plt.plot(x_vals, phi, 'o-', label= 'Numerical FD')
    plt.xlabel('x')
    plt.ylabel('ϕ(x)')
    plt.title('Numerical vs Analytical Solution of 1D Convection-Diffusion')
    plt.grid(True)
    plt.legend()
    plt.show()