from firedrake import *


# Setup parameters
L = 1.0
N = 32
dt = 0.01
T = 0.5
nu = 0.01

# Define mesh and function space
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "CG", 1)

# Define the Allen-Cahn equation
u = TrialFunction(V)
v = TestFunction(V)
F = (dot(grad(v), grad(u)) + u*(1 - u)*(u - 0.5)) * dx


# Define time dependent parameters
mu = Constant(nu)
f = Constant(0.0)

# Define initial conditions
u_n = Function(V)
u_n.interpolate(conditional(sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2) <= 0.2, 1.0, 0.0))


# Define the solver
solver = NonlinearVariationalSolver(problem, solver_parameters={'snes_type': 'newtonls'})

# Solve the problem
solver.solve()

# Save the solution
file = File("allen_cahn_solution.pvd")
file.write(u)