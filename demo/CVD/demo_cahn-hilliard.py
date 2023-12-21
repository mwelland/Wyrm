
# Begin demo

import random
from firedrake import *
import numpy as np

# Class representing the intial conditions
# class InitialConditions(Expression):
#     def __init__(self):
#         random.seed(2 + MPI.process_number())
#     def eval(self, values, x):
#         values[0] = 0.63 + 0.02*(0.5 - random.random())
#         values[1] = 0.0
#     def value_shape(self):
#         return (2,)

# Class for interfacing with the Newton solver
# class CahnHilliardEquation(NonlinearProblem):
#     def __init__(self, a, L):
#         NonlinearProblem.__init__(self)
#         self.L = L
#         self.a = a
#         self.reset_sparsity = True
#     def F(self, b, x):
#         assemble(self.L, tensor=b)
#     def J(self, A, x):
#         assemble(self.a, tensor=A, reset_sparsity=self.reset_sparsity)
#         self.reset_sparsity = False

# Model parameters
lmbda  = 1.0e-02  # surface parameter
dt     = 5.0e-06  # time step
theta  = 1      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

# Create mesh and define function spaces
mesh = SquareMesh(50, 50,1)
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

#*************************************************************************************
# Create INITIAL CONDITIONS and interpolate

# u.sub(0).interpolate(sin(x[0])/2+.5)
# u0.sub(0).interpolate(sin(x[0])/2+.5)
# u.sub(0).interpolate(-x[1]/2+.5)
# u0.sub(0).interpolate(-x[1]/2+.5)
# size_x = u.vector().size()
# print('size = ', size_x)

# print('arr =  ',arr)
# print(type(arr))
IC_arr = Function(V)
print('size = ',IC_arr.vector().size())
for i in range(IC_arr.vector().size()): 
    IC_arr.vector()[i] = 0.0#0.63 + 0.02*(0.5 - random.random())

#idx = int(IC_arr.vector().size()/2)
#IC_arr.vector()[idx] = 0.6
    
#Random IC
# u.sub(0).interpolate(x[1]-x[1]+0.5 + 0.02*( random.random()-0.5))
# u0.sub(0).interpolate(x[1]-x[1]+0.5+ 0.02*( random.random()-0.5))
#u.sub(0).interpolate(IC_arr)
#u0.sub(0).interpolate(IC_arr)

#Step function initial conditions
# u.sub(0).interpolate(1/(1+2.71**(-2.0*50.0*(x[1]-0.1))))
# u0.sub(0).interpolate(1/(1+2.71**(-2.0*50.0*(x[1]-0.1))))

#modified step function IC
u.sub(0).interpolate(1/(1+2.71**(-2.0*50.0*(x[1]-0.1)))*(x[1]**(0.1)))
u0.sub(0).interpolate(1/(1+2.71**(-2.0*50.0*(x[1]-0.1)))*(x[1]**(0.1)))
#*************************************************************************************

#boundary conditions
#bc1 = DirichletBC(ME.sub(0),0.0,3) # 3 for y=0 plane
#bc2 = DirichletBC(ME.sub(0),0.0,4) # 4 for y=1 plane
#bc3 = DirichletBC(ME.sub(0),1.0,1)
# Compute the chemical potential df/dc
c = variable(c)
double_well   = 100*c**2*(1-c)**2
f = double_well
dfdc = diff(f, c)

# mu_(n+theta)
mu_mid = (1.0-theta)*mu0 + theta*mu
c_mid = (1.0-theta)*c0 + theta*c

k = 1e-2
D = 2.0
D_c = D*(1/(1+2.71**(-2.0*3.0*(c-0.1))))**6
#n = FacetNormal(V)
# Weak statement of the equations
L0 = c*q*dx - c0*q*dx + D_c*dt*dot(grad(mu_mid), grad(q))*dx - k*q*ds(4)
L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
L = L0 + L1

# Compute directional derivative about u in the direction of du (Jacobian)
#a = derivative(L, u, du)

# Create nonlinear problem and Newton solver
# problem = CahnHilliardEquation(a, L)
# solver = NewtonSolver()
problem = NonlinearVariationalProblem(L,u)#,bcs=[bc1])
solver = NonlinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# Output file
file = File("output_stepic_Dc_d2_k1e-2.pvd", "compressed")

# Step in time
t = 0.0
T = 300*dt
while (t < T):
    t += dt
    #u0.vector()[:] = u.vector()
    u0.assign(u)
    solver.solve()
    #file << (u.split()[0], t)
    file.write(u.sub(0),time=t)

#plot(u.split()[0])
#interactive()
