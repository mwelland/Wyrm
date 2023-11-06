from firedrake import *
from tools import *
import thermo_potentials as tp
from math import log, ceil

from firedrake.petsc import PETSc
def print(*args, **kwargs):
    #Overloads print to be the petsc routine which relegates to the head mpi rank
    PETSc.Sys.Print(*args,flush=True)

M_phi = 1e-3#1e-8
D = 1e-3 #m^2/s = .1 cm^2/s
interface_width = .1

x_scale = 1
c_scale = 1

Lx = 10
Ly = Lx/1
Lz = Lx/1

mesh_res = interface_width

mesh = RectangleMesh(round(Lx/mesh_res), round(Ly/mesh_res), Lx/x_scale, Ly/x_scale)
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

pot = tp.load_potential('binary_2phase_elastic')

response = pot.grad([c_scale*cmesh[0], c_scale*cmesh[1]]+[p_phase, 1-p_phase])   #Fixme - shouldn't be negative

mu = as_vector(response[:n])
P = as_vector(response[n:])
#sigma = as_vector(response[2:8])

J =  -D*gr(mu)
F_diffusion = inner(J, gr(test_c))*dx
F_diffusion = 1/c_scale*F_diffusion

F_phase = -M_phi*inner(P, derivative(ps, phase, test_phase))*dx
F_phase += -M_phi*derivative(interface_energy*interface_area, phase, test_phase)*dx

F = F_diffusion + F_phase

params = {'snes_monitor': None,
          'snes_max_it': 10,
          'snes_atol':1e-6,
          'snes_rtol':1e-20,
          'pc_type': 'lu', 'ksp_type': 'preonly', 'pc_factor_mat_solver_type': 'mumps',
          }

# Since using a quadratic potential, we can just get initial values from expansion point
pt = pot.additional_fields['expansion_point']
print(pt)

ci_a = as_vector([pt['c0_a'], pt['c1_a']])/pt['V_a']
ci_b = as_vector([pt['c0_b'], pt['c1_b']])/pt['V_b']
print(ci_a)
print(ci_b)
# ci0 = as_vector([.2, .8])
# ci1 = as_vector([.8, .2])

# ~~~ Initial conditions ~~~ #
rc = 0*as_vector([1,1])
r = sqrt(inner(x-rc,x-rc))
#p0 = (.5*(1.-tanh((x[0]-.5*Lx)/(2.*interface_width))))# * (.5*(1.-tanh((3-x[0])/(2.*interface_width))))
p0 = (.5*(1.-tanh((r-.5*Lx)/(2.*interface_width))))# * (.5*(1.-tanh((3-x[0])/(2.*interface_width))))
#pp0 = p0**3*(6*p0**2-15*p0+10)

U.sub(1).interpolate(p0)

ic = p0*(1+0*1e-3)*ci_a+(1-p0)*ci_b
U.sub(0).interpolate(ic/c_scale)

# Boundary conditions
bcs = [
    #DirichletBC(V.sub(1), Constant(0), 2),
    #DirichletBC(V.sub(0), ci1/c_scale, 2),
    #DirichletBC(V.sub(3),Constant([0,0,0]), boundaries),
    ]


# Set up time stepper

dt = Constant(5.0e-2)
t_end = 10000.0
t = Constant(0.0)

tm = as_vector([1, 1, 1])
stepper = timestepper(U, test_U, [F_diffusion, F_phase], [], tm, bcs=bcs, dt = dt, params=params)

field_names = ['c', 'ps', 'P', 'mu']#, 'ca', 'cb']
writer = writer([ 'cmesh', 'phase'], field_names,[eval(f) for f in field_names],mesh)
writer.write(U, 0.0)

iter_t = 0
eps_tol_t= 10
eps_tol_t_target = eps_tol_t/2

phase_old = Function(V_phase)

while float(t) < t_end and iter_t<100:

    iter_t +=1
    phase_old.assign(U.sub(1))
    print('')

    print('{:n}: Time: {:6.4g}'.format(iter_t, float(t+dt)))
    try:
        so, eps_t, maxdt = stepper.step(dt)
        # If time step successful
        print('Succeeded. Estimated error: {:4.2g}, max change {:4.2g}'.format(eps_t, maxdt))
    except KeyboardInterrupt:
        print ('KeyboardInterrupt exception is caught')
        break

    except Exception as ex:
        # Time stepper failed. Retry with smaller timestep
        #raise
        print('failed')
        print(ex)
        dt.assign(float(dt)*.5)
        stepper.reset_step()
        continue

    # Time step was successful and has been accepted
    t.assign(float(t)+float(dt))
    stepper.accept_step()

    # Adapt the time step to some metric
    dphase = errornorm(phase,phase_old,'l10')
    print('max phase change', dphase)
    dphase_target = .1
    #dt.assign(float(dt)*min( (eps_tol_t_target/(eps_t+1e-10)), 5))  # Change timestep to aim for tolerance
    dt.assign(float(dt)*min( (dphase_target/(dphase)), 20))  # Change timestep to aim for tolerance

    if iter_t %1 ==0:
        writer.write(U, time=float(t))
