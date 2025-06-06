from firedrake import *
from tools import *
#from ufl_tools import max_value
from thermo_potentials import load_potential
from math import log, ceil

M_phi = 1e-3#1e-8
D = 1e-3 #m^2/s = .1 cm^2/s
interface_width = .1

x_scale = 1
c_scale = 1

Lx = 2
Ly = Lx/1
Lz = Lx/1

# Coarse mesh should have an 'appreciable' resolution. Fine mesh is scale of feature of interest
mesh_res_coarse = Lx/4
mesh_res_final = interface_width #target mesh resolution
mg_levels = ceil( log(mesh_res_coarse/mesh_res_final,2) )
print('Using {} levels of refinement'.format(mg_levels))

mesh = BoxMesh(round(Lx/mesh_res_coarse), round(Ly/mesh_res_coarse), round(Lz/mesh_res_coarse), Lx/x_scale, Ly/x_scale, Lz/x_scale, reorder=True)
#mesh = BoxMesh(round(Lx/mesh_res_final), round(Ly/mesh_res_final), round(Lz/mesh_res_final), Lx/x_scale, Ly/x_scale, Lz/x_scale, reorder=True)

#mesh = RectangleMesh(round(Lx/mesh_res_coarse), round(Ly/mesh_res_coarse), Lx/x_scale, Ly/x_scale)

hierarchy = MeshHierarchy(mesh, mg_levels)
mesh = hierarchy[-1]

# utility function to help with non-dimensionalization
def gr(x):
    return grad(x)/x_scale

#n - number of species, m = number of phases
n = 2
m = 3

xmesh = SpatialCoordinate(mesh)
x = xmesh*x_scale

V_phase = VectorFunctionSpace(mesh, "CG", 1, dim = m-1, name="phases")
V_species = VectorFunctionSpace(mesh, "CG", 1, dim=n, name ="species")
V = MixedFunctionSpace([V_species, V_phase])

U = Function(V)
dU = TrialFunction(V)
test_U = TestFunction(V)
test_c, test_phase = split(test_U)

cmesh, phase = split(U)

c = c_scale*cmesh

#Assemble full vector of phi and p_phase
phi = [p for p in phase]+[1-sum(phase)]
p_phase = [p**3*(6*p**2-15*p+10) for p in phi]
ps = as_vector(p_phase)

# Build multiphase energy -> to be moved to thermo potential.
def multiphase(p, interface_width):
    def antisymmetric_gradient(pa, pb):
        return 3*(interface_width**2*( pa*gr(pb) - pb*gr(pa) )**2 + pa**2*pb**2)
    return [antisymmetric_gradient(p[i], p[j]) for i in range(len(p)) for j in range(i)]
interface_area =  multiphase(phi, interface_width)

# pa = phi[0]
# pb = phi[1]
# pc = phi[2]
# a = 50
# interface_energy = 5000*3*(interface_width**2*( pa*gr(pb) - pb*gr(pa) )**2 + pa**2*pb**2*(1+a*pc**2)
#                     + interface_width**2*( pc*gr(pb) - pb*gr(pc) )**2 + pc**2*pb**2*(1+a*pa**2)
#                     + interface_width**2*( pc*gr(pa) - pa*gr(pc) )**2 + pc**2*pa**2*(1+a*pb**2))

interface_energy = inner(as_vector([5000]*m), as_vector(interface_area))

#Load potential
pot = load_potential('binary_3phase_elastic')
response = pot.grad([ci for ci in c]+p_phase)
mu = as_vector(response[:n])
P = as_vector(response[n:])
print('Thermodynamic driver forces loaded')

# build diffusion equation
J =  -D*gr(mu)
F_diffusion = inner(J, gr(test_c))*dx
F_diffusion = 1/c_scale*F_diffusion

# build phase field equaiton
F_phase_bulk = -M_phi*inner(P, derivative(ps, phase, test_phase))*dx
F_phase_interface = -M_phi*derivative( interface_energy, phase, test_phase)*dx
F_phase = F_phase_bulk + F_phase_interface

# ~~~ Initial conditions ~~~ #
# phase initial conditions
def create_bubble(centre, radius):
    centre = as_vector(centre)
    r = sqrt(inner(x-centre, x-centre))
    return .5*(1.-tanh((r-radius)/(2.*interface_width)))
p0 = create_bubble( [.2*Lx, .2*Lx, .2*Lx], .4*Lx)
p1 = create_bubble( [.8*Lx, .8*Lx, .8*Lx], .4*Lx)

U.sub(1).interpolate(as_vector([p0,p1]))

# Since using a quadratic potential, we can just get initial values from expansion point
pt = pot.additional_fields['expansion_point']
ci = as_matrix([
    [pt['c0_a']/pt['V_a'], pt['c1_a']/pt['V_a']],
    [pt['c0_b']/pt['V_b'], pt['c1_b']/pt['V_b']],
    [pt['c0_c']/pt['V_c'], pt['c1_c']/pt['V_c']]])
U.sub(0).interpolate(dot(ps,ci)/c_scale)

# Boundary conditions
bcs = [
    DirichletBC(V.sub(1).sub(0), Constant(0), 2),
    #DirichletBC(V.sub(0), ci1/c_scale, 2),
    ]

### ~~~ Set-up solver and timestepping
params = {'snes_monitor': None,
          'snes_max_it': 10,
          'snes_atol':1e-6,
          'snes_rtol':1e-20,
          'snes_view': None,
          'ksp_converged_reason': None,
          #'snes_linesearch_type': 'bt',

          #Direct
          #'pc_type': 'lu', 'ksp_type': 'preonly', 'pc_factor_mat_solver_type': 'mumps',

          #Geometric multigrid
          'ksp_type':'fgmres', 'pc_type':'mg', 'mg_coarse_pc_type':'lu','mg_coarse_pc_factor_mat_solver_type':'mumps',

          }

scheme = time_stepping_scheme(U, test_U, [F_diffusion, F_phase], [],
    time_coefficients = as_vector([1,1,1,1]),
    bcs = bcs,
    params = params)

field_names = ['c', 'ps', 'P', 'mu']
writer = writer([ 'cmesh', 'phase'], field_names, [eval(f) for f in field_names], mesh)

solve_time_series(scheme, writer,
    t_range = [0, 5e-2, 1e4],
    iter_t_max = 10,
    eps_t_target = .1,
    eps_s_target = .2)
