from firedrake import *
from tools import *
from thermo_potentials import load_potential
from math import log, ceil
import random

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
mesh_res_final = interface_width/2 #target mesh resolution
mg_levels = ceil( log(mesh_res_coarse/mesh_res_final,2) )
print('Using {} levels of refinement'.format(mg_levels))

mesh = BoxMesh(round(Lx/mesh_res_coarse), round(Ly/mesh_res_coarse), round(Lz/mesh_res_coarse), Lx/x_scale, Ly/x_scale, Lz/x_scale, reorder=True)

hierarchy = MeshHierarchy(mesh, mg_levels)
mesh = hierarchy[-1]
print('Mesh hierarchy assembled')

# utility function to help with non-dimensionalization
def gr(x):
    return grad(x)/x_scale

#n - number of species, m = number of phases
n = 2
m = 2

xmesh = SpatialCoordinate(mesh)
x = xmesh*x_scale

V_phase = FunctionSpace(mesh, "CG", 1, name="phases")
V_species = VectorFunctionSpace(mesh, "CG", 1, dim=n, name ="species")
V = MixedFunctionSpace([V_species, V_phase])

U = Function(V)
dU = TrialFunction(V)
test_U = TestFunction(V)
test_c, test_phase = split(test_U)

cmesh, phase = split(U)
c = c_scale*cmesh

# Phase field functions
p_phase = phase**3*(6*phase**2-15*phase+10)
g_phase = phase**2*(1-phase)**2
interface_area = 3*( interface_width**2*inner(gr(phase),gr(phase)) + g_phase)
interface_energy = 5000

ps = as_vector([p_phase, 1-p_phase])

# Load potential
#pot = load_potential('binary_and_stoichiometric_2phase_elastic')
pot = load_potential('SiC_potential')
response = pot.grad([c_scale*cmesh[0], c_scale*cmesh[1]]+[p_phase, 1-p_phase])   #Fixme - shouldn't be negative

mu = as_vector(response[:n])
P = as_vector(response[n:])
print('Thermodynamic driver forces loaded')

J =  -D*gr(mu)
F_diffusion = inner(J, gr(test_c))*dx
F_diffusion = 1/c_scale*F_diffusion

F_phase_bulk = -M_phi*inner(P, derivative(ps, phase, test_phase))*dx
F_phase_interface = -M_phi*derivative(interface_energy*interface_area, phase, test_phase)*dx
F_phase = F_phase_bulk + F_phase_interface

params = {'snes_monitor': None,
          'snes_max_it': 10,
          'snes_atol':1e-6,
          'snes_rtol':1e-20,
          'snes_view': None,
          'ksp_converged_reason': None,
          #'snes_linesearch_type': 'bt',

          #Direct
          'pc_type': 'lu', 'ksp_type': 'preonly', 'pc_factor_mat_solver_type': 'mumps',

          #Geometric multigrid
          #'ksp_type':'fgmres', 'pc_type':'mg', 'mg_coarse_pc_type':'lu','mg_coarse_pc_factor_mat_solver_type':'mumps',
          }



# ~~~ Initial conditions ~~~ #

p_surface = 1/(1+2.71**(2.0*50.0*(x[2]-0.1)))#*(x[2]**(0.1))


r = 0.1*Lx #radius of bubble
arr_centres = define_centres_arr(0,1,0.2,Lx,3,False,0.1)
p_bubbles = sum_bubbles(arr_centres,r,x,interface_width)

p = max_value(p_bubbles,p_surface)
U.sub(1).interpolate(p)

# Since using a quadratic potential, we can just get initial values from expansion point
pt = pot.additional_fields['expansion_point']
ci = as_matrix([
    [pt['c0_a']/pt['V_a'], 0],
    [pt['c0_b']/pt['V_b'], pt['c1_b']/pt['V_b']]])
U.sub(0).interpolate(dot(ps,ci)/c_scale)


# Boundary conditions
bcs = [
    DirichletBC(V.sub(1), Constant(0), 6),
    DirichletBC(V.sub(0),as_vector([0.9,0.2]),6),
    #DirichletBC(V.sub(0), ci1/c_scale, 2),
    #DirichletBC(V.sub(3),Constant([0,0,0]), boundaries),
    ]


# Set up time stepper

scheme = time_stepping_scheme(U, test_U, [F_diffusion, F_phase], [],
    time_coefficients = as_vector([1,1,1]),
    bcs = bcs,
    params=params)

field_names = ['c', 'ps', 'P', 'mu']#, 'ca', 'cb']
writer = writer([ 'cmesh', 'phase'], field_names,[eval(f) for f in field_names],mesh,"output_SiC_2phase_3d/output.pvd")

solve_time_series(scheme, writer,
    t_range = [0, 5e-2, 1e4],
    iter_t_max = 250,
    eps_t_target = .1,
    eps_s_target = .2)
