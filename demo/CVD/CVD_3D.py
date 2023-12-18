import random
from firedrake import *
import numpy as np


# Model parameters
lmbda  = 1.0e-02  # surface parameter
dt     = 5.0e-06  # time step
theta  = 1      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson
Lx = 1
Ly = Lx/1
Lz = Lx/1

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Create mesh and define function spaces
mesh = BoxMesh(50,50,50,Lx,Ly,Lz)
V = FunctionSpace(mesh, "Lagrange", 1)
ME = V*V
x = SpatialCoordinate(mesh)
# Define trial and test functions
du    = TrialFunction(ME)
q, v  = TestFunctions(ME)

# Define functions
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step

# Split mixed functions
dc, dmu = split(du)
c,  mu  = split(u)
c0, mu0 = split(u0)

# Create intial conditions and interpolate
# u_init = InitialConditions()
# u.interpolate(u_init)
# u0.interpolate(u_init)
# u.sub(0).interpolate(sin(x[0])/2+.5)
# u0.sub(0).interpolate(sin(x[0])/2+.5)
u.sub(0).interpolate(-x[1]/2+.5)
u0.sub(0).interpolate(-x[1]/2+.5)

#boundary conditions
bc1 = DirichletBC(ME.sub(0),1.0,5) # 3 for y=0 plane
bc2 = DirichletBC(ME.sub(0),0.0,6) # 4 for y=1 plane


# Compute the chemical potential df/dc
c = variable(c)
double_well   = 100*c**2*(1-c)**2
f = double_well
dfdc = diff(f, c)

# mu_(n+theta)
mu_mid = (1.0-theta)*mu0 + theta*mu

# Weak statement of the equations
L0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
L = L0 + L1

# Compute directional derivative about u in the direction of du (Jacobian)
#a = derivative(L, u, du)

# Create nonlinear problem and Newton solver
# problem = CahnHilliardEquation(a, L)
# solver = NewtonSolver()
problem = NonlinearVariationalProblem(L,u,bcs=[bc1,bc2])
solver = NonlinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# Output file
file = File("output3d.pvd", "compressed")

# Step in time
t = 0.0
T = 50*dt
while (t < T):
    t += dt
    #u0.vector()[:] = u.vector()
    u0.assign(u)
    solver.solve()
    #file << (u.split()[0], t)
    file.write(u.sub(0),time=t)

#plot(u.split()[0])
#interactive()
