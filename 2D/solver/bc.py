class DirichletBC:
    def __init__(self, analytical_solution):
        self.analytical_solution = analytical_solution
        self.type = "dirichlet"

    def value(self, i, j):
        x = i / (self.nx - 1)
        y = j / (self.ny - 1)
        return self.analytical_solution(x, y)

    def set_grid(self, nx, ny):
        self.nx = nx
        self.ny = ny


class NeumannBC:
    def __init__(self, derivative_value=0.0):
        self.derivative_value = derivative_value
        self.type = "neumann"

    def set_grid(self, nx, ny):
        self.nx = nx
        self.ny = ny
