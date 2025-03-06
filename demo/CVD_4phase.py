from firedrake import *
from tools import *
#from ufl_tools import max_value
from thermo_potentials import load_potential
from math import log, ceil, comb

M_phi = 1e-3#1e-8

interface_width = .1*2

x_scale = 1
c_scale = 1

#Lx = 10
Lx = 5 #interface_width*60 
Ly = Lx/1
Lz = 4

# Coarse mesh should have an 'appreciable' resolution. Fine mesh is scale of feature of interest
mesh_res_coarse = Lx/4
mesh_res_final = interface_width/2 #target mesh resolution
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
m = 4

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
        return 3*(interface_width**2*( pa*gr(pb) - pb*gr(pa) )**2 + pa**2*pb**2*(1+50*(pa+pb-1)**2))
    return [antisymmetric_gradient(p[i], p[j]) for i in range(len(p)) for j in range(i)]
interface_area =  multiphase(phi, interface_width)

# pa = phi[0]
# pb = phi[1]
# pc = phi[2]
# a = 50
# interface_energy = 5000*3*(interface_width**2*( pa*gr(pb) - pb*gr(pa) )**2 + pa**2*pb**2*(1+a*pc**2)
#                     + interface_width**2*( pc*gr(pb) - pb*gr(pc) )**2 + pc**2*pb**2*(1+a*pa**2)
#                     + interface_width**2*( pc*gr(pa) - pa*gr(pc) )**2 + pc**2*pa**2*(1+a*pb**2))

interface_energy = inner(as_vector([1e5]*comb(m,2)), as_vector(interface_area))

#Load potential
pot = load_potential('CVD_4phase_pot')

response = pot.grad([ci for ci in c]+p_phase)
mu = as_vector(response[:n])
P = as_vector(response[n:])
print('Thermodynamic driver forces loaded')

# build diffusion equation
#D = 1e-3 #m^2/s = .1 cm^2/s
D = inner(as_vector(phi),as_vector([1e-3,1e-3,1e-3,1e-3]))
J =  -D*gr(mu)
F_diffusion = inner(J, gr(test_c))*dx
F_diffusion = 1/c_scale*F_diffusion

# build phase field equaiton
F_phase_bulk = -M_phi*inner(P, derivative(ps, phase, test_phase))*dx
F_phase_interface = -M_phi*derivative( interface_energy, phase, test_phase)*dx
F_phase = F_phase_bulk + F_phase_interface

# ~~~ Initial conditions ~~~ #
# phase initial conditions
rad = interface_width*2
rad2 = interface_width*4
rad3 = interface_width*3
rad4 = interface_width*9

a1_centres = [[0.25*Lx,0.25*Lx,0],[0.5*Lx,0.5*Lx,0],[0.75*Lx,0.75*Lx+2.5*rad,0],[0.75*Lx-rad2-3*rad,0.25*Lx+rad,0],
              [0.65*Lx,0.645*Lx,0],[0.45*Lx,0.16*Lx,0],[0.35*Lx,0.4*Lx,0],[0.5*Lx,0.3*Lx,0]
              ]
a1 = create_bubble( a1_centres, rad,x, interface_width)

a2_centres = [[0.25*Lx+rad2,0.75*Lx-rad2,0],[0.75*Lx,0.25*Lx,0]
              ]
a2 = create_bubble( a2_centres, rad2,x, interface_width)

a3_centres = [[Lx-1.1*rad3,0.6*Lx,0]
              ]
a3 = create_bubble( a3_centres, rad3,x, interface_width)


b1_centres = [[0.5*Lx+rad/.7,0.5*Lx+2*rad,0],[0.75*Lx+rad/.7,0.75*Lx+rad,0],[0.75*Lx-rad2-rad,0.25*Lx,0],
              [0.25*Lx+rad/0.7,0.25*Lx+2*rad,0],[0.52*Lx,0.1*Lx,0],[0.66*Lx,0.43*Lx,0],[0.48*Lx,0.3*Lx-2*rad,0]
              ]
b1 = create_bubble( b1_centres, rad,x, interface_width)

b2_centres = [[0.25*Lx+rad2/1.4,0.75*Lx+rad2,0],[Lx-rad2,0.25*Lx+1.25*rad2,0],[1.2*rad2,1.2*rad2,0]
              ]
b2 = create_bubble( b2_centres, rad2,x, interface_width)

b3_centres = [[0.25*Lx-rad3,0.5*Lx,0]
              ]
b3 = create_bubble( b3_centres, rad3,x, interface_width)

b4_centres = [[0.5*Lx-2*rad,0.5*Lx,0],[0.65*Lx,0.65*Lx+2*rad,0]
              ]
b4 = create_bubble( b4_centres, rad,x, interface_width)


c1_centres = [[0.5*Lx+rad/0.7,0.5*Lx-2*rad,0],[0.75*Lx+2*rad,0.75*Lx+3*rad,0],[0.25*Lx+rad,0.5*Lx,0],
              [0.75*Lx-rad2-rad/1.4,0.25*Lx+2*rad,0],[0.5*Lx-rad/1.4,0.5*Lx+2*rad,0],[0.75*Lx,0.48*Lx,0],
              [1.5*rad,0.25*Lx,0],[0.25*Lx-2*rad,0.25*Lx+2*rad,0],[0.48*Lx,0.3*Lx+2*rad,0]
              ]
c1 = create_bubble( c1_centres, rad,x, interface_width)


c2_centres = [[0.75*Lx-2*rad2,0.75*Lx+rad2,0],[1.1*rad2,0.8*Lx,0]
              ]
c2 = create_bubble( c2_centres, rad2,x, interface_width)

c3_centres = [[0.35*Lx,0.15*Lx,0],[0.65*Lx,0.1*Lx,0],[0.85*Lx,0.15*Lx,0]
              ]
c3 = create_bubble( c3_centres, rad,x, interface_width)

# numart
# a1_centres = [[0.25*Lx,3.5*rad,0],[0.5*Lx,0.5*Ly,0],[Lx-2*rad,rad,0],[0.8*Lx,0.65*Ly,rad],
#               [0.65*Lx,0.645*Ly,0],[0.45*Lx,0.16*Ly,0],[0.35*Lx,0.9*rad,0.4*rad],[0.5*Lx,0.3*Ly,0],
#               [rad,rad,2*rad],[rad,rad,3*rad],[4*rad,0.5*rad,0],[4*rad,0.5*rad,rad],
#               [0.25*Lx+rad,rad,0.5*Lz-2*rad2-3*rad],[0.75*Lx-2*rad,rad,0.5*rad],
#               [0.75*Lx+5*rad,rad,0.25*Lz],[0.75*Lx+5*rad,rad,0.25*Lz+rad],
#               [0.75*Lx+2*rad,rad,3*rad],[0.58*Lx,rad,0.2*Lz+2*rad],[0.5*Lx,rad,1.75*Lz],
#               [0.45*Lx,rad,1.5*Lz],[0.3*Lx,0.7*rad,0.25*Lz],[0.18*Lx,0.1*rad,0.2*Lz],
#               [0.25*Lx,0,5.2*rad],

#               #c4
#               [0.48*Lx,3*rad,0.25*Lz],[rad,3*rad,0.25*Lz],[rad,3*rad,0.25*Lz-rad],
#               [0.75*Lx,3*rad,1.5*Lz],[1.2*Lx,3.5*rad,0.1*Lz],
#               [1.2*Lx,3.5*rad,0.1*Lz+rad],[0.8*Lx,3*rad,0.4*Lz],[3*rad,4*rad,1.5*Lz],
#               [3*rad,4*rad,1.5*Lz+rad],[0.1*Lx,3*rad,0.1*Lz],


#               [rad,Ly-rad,0.4*Lz],[3*rad,Ly-3*rad,0.4*Lz],[0.45*Lx,Ly-0.8*rad,0.4*Lz],
#               [rad,Ly-rad,0.4*Lz-rad],[3*rad,Ly-3*rad,0.4*Lz-rad],[0.45*Lx,Ly-0.8*rad,0.4*Lz-rad]
#               ]
# a1 = create_bubble( a1_centres, rad,x, interface_width)

# a2_centres = [[0.25*Lx+rad2,rad2,0.5*Lz],[0.75*Lx,0.9*rad2,0.6*Lz],
#               [0.75*Lx,0.9*rad2,0.6*Lz+0.8*rad2],

#               [0.4*Lx,3*rad2,0.5*Lz-2*rad2],[0.75*Lx+2*rad2,3*rad2,0.5*Lz-rad],
#               [0.3*Lx,3*rad2,0.5*Lz+2*rad2],

#               [0.62*Lx,Ly-0.8*rad3,0.62*Lz],[0.61*Lx,Ly-0.8*rad3,0.62*Lz+0.8*rad3]
#               ]
# a2 = create_bubble( a2_centres, rad2,x, interface_width)

# a3_centres = [[1.75*Lx,rad,0.35*Lz],[0.5*Lx+rad,rad,0.5*Lz],[0.7*Lx,rad,4.5*rad],
#               [rad,rad,0.5*Lz],[0.5*rad,rad,0.5*Lz-rad],[0.5*Lx,rad,0.6*Lz],
#               [0.5*Lx,rad,0.6*Lz+rad],[0.7*Lx,rad,0.35*Lz],
#               [2.8*rad,rad,0.25*Lz+2.2*rad],[2.8*rad,rad,0.25*Lz+1.2*rad],
#               [0.45*Lz,2*rad,1.6*Lz],[0.2*Lx,rad,0.4*Lz],[0.4*Lx,0.8*rad,2*Lz],
#               [0.4*Lx,0.8*rad,0.16*Lz],[Lx-rad,0.7*rad,0.11*Lz],
#               [0.5*Lx+0.5*rad,rad,3*rad],[0.5*Lx+0.5*rad,rad,2*rad],
#               [0.8*Lx,0.5*rad,0.3*Lz],

#               [0.3*Lx,.45*Ly,0.25*Lz],[0.3*Lx,.45*Ly,0.25*Lz+rad],
#               [0.53*Lx,.45*Ly,0.25*Lz],[0.53*Lx,.45*Ly,0.25*Lz+rad]
#               ]

# a3 = create_bubble( a3_centres, rad,x, interface_width)


# b1_centres = [[0.5*Lx+2*rad,rad,0],[0.75*Lx,rad,0],[0,rad,0],[Lx-2*rad,0.65*Ly,0],
#               [0.25*Lx+rad/0.7,rad,2*rad],[0.25*Lx+rad/0.7,rad,3*rad],[2*rad,Ly-rad,0],
#               [0.75*Lx,rad,0.25*Lz],[0.75*Lx,rad,0.25*Lz+rad],
#               [0.25*Lx-2*rad,rad,0.25*Lz],[0.38*Lx,0.5*Ly,rad],
#               [0.55*Lx,rad,0.2*Lz],[0.7*Lx,rad,2.5*rad],[2.5*rad,rad,0.45*Lz],[0.6*Lx,rad,0.2*Lz],
#               [0.46*Lx,rad,1.5*Lz],[0.2*Lx,rad,1.5*Lz],
#               [0.2*Lx,rad,1.5*Lz+rad],[0.55*Lx,rad,0.4*Lz],[0.1*Lx+2*rad,rad,0.1*Lz],
#               [0.1*Lx+2*rad,rad,0.1*Lz+rad],[0.2*Lx+2*rad,rad,0.4*Lz],[0.8*rad,0.8*rad,0.32*Lz],
#               [0.46*Lx,0.8*rad,0.1*Lz],[Lx-rad,0.7*rad,1.6*Lz],
#               [0.9*Lx,0.6*rad,0.15*Lz],[Lx-3*rad,0.5*rad,0.2*Lz],

#                 #a1
#               [0.25*Lx,3.5*rad+2*rad,0],[0.5*Lx,0.5*Ly+2*rad,0],[Lx-2*rad,rad+2*rad,0],[0.8*Lx,0.65*Ly+2*rad,rad],
#               [0.65*Lx,0.645*Ly+2*rad,0],[0.45*Lx,0.16*Ly+2*rad,0],[0.35*Lx,0.9*rad+2*rad,0.4*rad],[0.5*Lx,0.3*Ly+2*rad,0],
#               [rad,rad+2*rad,2*rad],[rad,rad+2*rad,3*rad],[4*rad,0.5*rad+2*rad,0],[4*rad,0.5*rad+2*rad,rad],
#               [0.25*Lx+rad,rad+2*rad,0.5*Lz-2*rad2-3*rad],[0.75*Lx-2*rad,rad+2*rad,0.5*rad],
#               [0.75*Lx+5*rad,3*rad,0.25*Lz],[0.75*Lx+5*rad,3*rad,0.25*Lz+rad],
#               [0.75*Lx+2*rad,3*rad,3*rad],[0.58*Lx,3*rad,0.2*Lz+2*rad],[0.5*Lx,rad+2*rad,1.75*Lz],
#               [0.45*Lx,rad+2*rad,1.5*Lz],[0.3*Lx,0.7*rad+2*rad,0.25*Lz],[0.18*Lx,0.1*rad+2*rad,0.2*Lz],
#               [0.25*Lx,2*rad,5.2*rad],

#               [0.25*Lx,Ly-rad,0],[0.5*Lx,Ly-rad,0],[Lx-2*rad,Ly-rad,0],[0.8*Lx,Ly-rad,rad],
#               [0.65*Lx,Ly-rad,0],[0.45*Lx,Ly-rad,0],[0.35*Lx,Ly-0.9*rad,0.4*rad],[0.5*Lx,Ly-rad,0],
#               [rad,Ly-rad,0.25*Lz],[rad,Ly-rad,0.25*Lz+rad],[4*rad,Ly-0.5*rad,0],[4*rad,Ly-rad,rad]
              
#               ,
#               [rad,0.5*Ly,0.2*Lz],[3*rad,0.5*Ly,0.2*Lz+rad],[4*rad,0.5*Ly,0.2*Lz],
#               [0.73*Lx,.45*Ly,0.25*Lz],[0.73*Lx,.45*Ly,0.25*Lz-rad]
#               ]
# b1 = create_bubble( b1_centres, rad,x, interface_width)

# b2_centres = [[0.4*Lx,rad2,0.5*Lz-2*rad2],[0.75*Lx+2*rad2,rad2,0.5*Lz-rad],
#               [0.3*Lx,rad2,0.5*Lz+2*rad2],
#               [0.15*Lx,3.1*rad2,0.65*Lz-rad2],[0.65*Lx,3*rad3,0.5*Lz-rad],
#               ]
# b2 = create_bubble( b2_centres, rad2,x, interface_width)

# b3_centres = [[0.6*Lx,rad3,0.6*Lz],[0.3*Lx+rad,0.8*rad3,0.5*Lz+2.8*rad2],
#                [0.6*Lx,0.8*rad3,0.6*Lz+rad],

#                [0.9*Lx,3*rad3,0.6*Lz],[0.15*Lx-0.5*rad,3.8*rad3,0.65*Lz],

#                [0.45*Lx,Ly-rad3,0.6*Lz],[0.45*Lx,Ly-rad3,0.6*Lz+0.8*rad3]
#               ]
# b3 = create_bubble( b3_centres, rad3,x, interface_width)



# c1_centres = [[Lx,rad,0],[0.75*Lx+2*rad,rad,0],[2*rad,rad,0],[Lx-2*rad,0.5*Ly-rad,2*rad],
#               [0.25*Lx,rad,0],[0.5*Lx+2.5*rad,rad,3*rad],[0.5*Lx+2.5*rad,rad,2*rad],
#               [0.25*Lx+3.5*rad,rad,2*rad],[0.25*Lx,rad2,0.5*Lz-2*rad2],
#               [0.25*Lx,rad2,0.5*Lz-2*rad2-rad],[0.75*Lx+2*rad,2*rad,0.25*Lz],
#               [0.75*Lx+2*rad,2*rad,0.25*Lz-rad],[0.5*Lx-rad,rad,0.5*Lz],
#               [0.3*Lx+2*rad,0.75*rad,2.2*Lz],[0.66*Lx,0.8*rad,0.28*Lz],
#               [0.45*Lx-0.5*rad,0.8*rad,0.65*Lz],[0.45*Lx,0.8*rad,0.65*Lz+rad],
#               [0.25*Lx,0.8*rad,0.31*Lz],[0.83*Lx,0.8*rad,0.2*Lz],
#               [0.25*Lx,0.7*rad,rad],[0.25*Lx,0.5*rad,0.1*Lz],[0.35*Lx,0.5*rad,0.2*Lz],
#               [Lx-0.8*rad,0.8*rad,0.36*Lz],[Lx-0.8*rad,0.8*rad,0.18*Lz],
#                 #a3
#               [1.75*Lx,3*rad,0.35*Lz],[0.5*Lx+rad,3*rad,0.5*Lz],[0.7*Lx,3*rad,4.5*rad],
#               [rad,3*rad,0.5*Lz],[0.5*rad,3*rad,0.5*Lz-rad],[0.5*Lx,3*rad,0.6*Lz],
#               [0.5*Lx,3*rad,0.6*Lz+rad],[0.7*Lx,3*rad,0.35*Lz],
#               [2.8*rad,3*rad,0.25*Lz+2.2*rad],[2.8*rad,3*rad,0.25*Lz+1.2*rad],
#               [0.45*Lz,4*rad,1.6*Lz],[0.2*Lx,3*rad,0.4*Lz],[0.4*Lx,2.8*rad,2*Lz],
#               [0.4*Lx,2.8*rad,0.16*Lz],[Lx-rad,2.7*rad,0.11*Lz],
#               [0.5*Lx+0.5*rad,3*rad,3*rad],[0.5*Lx+0.5*rad,3*rad,2*rad],
#               [0.8*Lx,2.5*rad,0.3*Lz],

#                 #b1
#               [0.5*Lx+2*rad,3*rad,0],[0.75*Lx,3*rad,0],[0,3*rad,0],[Lx-2*rad,0.65*Ly+2*rad,0],
#               [0.25*Lx+rad/0.7,3*rad,2*rad],[0.25*Lx+rad/0.7,3*rad,3*rad],[2*rad,Ly-3*rad,rad],
#               [0.75*Lx,3*rad,0.25*Lz],[0.75*Lx,3*rad,0.25*Lz+rad],
#               [0.25*Lx-2*rad,3*rad,0.25*Lz],[0.38*Lx,0.5*Ly+2*rad,rad],
#               [0.55*Lx,3*rad,0.2*Lz],[0.7*Lx,3*rad,2.5*rad],[2.5*rad,3*rad,0.45*Lz],[0.6*Lx,3*rad,0.2*Lz],
#               [0.46*Lx,3*rad,1.5*Lz],[0.2*Lx,3*rad,1.5*Lz],
#               [0.2*Lx,3*rad,1.5*Lz+rad],[0.55*Lx,3*rad,0.4*Lz],[0.1*Lx+2*rad,3*rad,0.1*Lz],
#               [0.1*Lx+2*rad,3*rad,0.1*Lz+rad],[0.2*Lx+2*rad,3*rad,0.4*Lz],[0.8*rad,2.8*rad,0.32*Lz],
#               [0.46*Lx,2.8*rad,0.1*Lz],[Lx-rad,2.7*rad,1.6*Lz],
#               [0.9*Lx,2.6*rad,0.15*Lz],[Lx-3*rad,2.5*rad,0.2*Lz],


#               [0.25*Lx,Ly-rad,0],[0.5*Lx,Ly-rad,0],[Lx-2*rad,Ly-rad,0],[0.8*Lx,Ly-rad,rad],
#               [0.65*Lx,Ly-rad,0],[0.45*Lx,Ly-rad,0],[0.35*Lx,Ly-rad,0.4*rad],[0.5*Lx,Ly-rad,0],
#               [rad,Ly-rad,2*rad],[rad,Ly-rad,3*rad],[4*rad,Ly-rad,0],[4*rad,Ly-rad,rad],
#               [0.25*Lx+rad,Ly-rad,0.5*Lz-2*rad2-3*rad],[0.75*Lx-2*rad,Ly-rad,0.5*rad],
#               [0.75*Lx+5*rad,Ly-rad,0.25*Lz],[0.75*Lx+5*rad,Ly-rad,0.25*Lz+rad],
#               [0.75*Lx+2*rad,Ly-rad,3*rad],[0.58*Lx,Ly-rad,0.2*Lz+2*rad],[0.5*Lx,Ly-rad,1.75*Lz],
#               [0.45*Lx,Ly-rad,1.5*Lz],[0.3*Lx,Ly-rad,0.25*Lz],[0.18*Lx,Ly-rad,0.2*Lz],
#               [0.25*Lx,Ly-rad,5.2*rad],

#               [0.5*Lz,Ly-rad,0.35*Lz],[0.5*Lz,Ly-rad,0.35*Lz+rad]
#               ]
# c1 = create_bubble( c1_centres, rad,x, interface_width)

# c2_centres = [[0.15*Lx,rad2,0.65*Lz-rad2],[0.65*Lx,rad3,0.5*Lz-rad],
#               [0.25*Lx+rad2,3*rad2,0.5*Lz],[0.75*Lx,2.9*rad2,0.6*Lz],
#               [0.75*Lx,2.9*rad2,0.6*Lz+0.8*rad2]
#               ]
# c2 = create_bubble( c2_centres, rad2,x, interface_width)

# c3_centres = [[0.9*Lx,rad3,0.6*Lz],[0.15*Lx-0.5*rad,0.8*rad3,0.65*Lz],
#               [0.49*Lx,Ly-3.1*rad3,0.6*Lz],[0.49*Lx,Ly-3.1*rad3,0.6*Lz+0.8*rad3]
#               ]
# c3 = create_bubble( c3_centres, rad3,x, interface_width)

# c4_centres = [[0.48*Lx,rad,0.25*Lz],[rad,rad,0.25*Lz],[rad,rad,0.25*Lz-rad],
#               [0.75*Lx,rad,1.5*Lz],[1.2*Lx,1.5*rad,0.1*Lz],
#               [1.2*Lx,1.5*rad,0.1*Lz+rad],[0.8*Lx,rad,0.4*Lz],[3*rad,2*rad,1.5*Lz],
#               [3*rad,2*rad,1.5*Lz+rad],[0.1*Lx,rad,0.1*Lz]
#               ]
# c4 = create_bubble( c4_centres, rad,x, interface_width)

# ######



# a1_centres = [[rad,rad,0],[Lx-rad,Lx-rad,0],[0.5*Lx+2.1*rad,0.5*Lx,0]
#               ]
# a1 = create_bubble( a1_centres, rad,x, interface_width)

# b1_centres = [[Lx-rad,rad,0],[0.5*Lx,0.5*Lx,0]
#               ]
# b1 = create_bubble( b1_centres, rad,x, interface_width)

# c1_centres = [[rad,Ly-rad,0],[0.5*Lx,0.75*Lx,0]
#               ]
# c1 = create_bubble( c1_centres, rad,x, interface_width)
# c2_centres = [[0.5*Lx,0.8*rad2,0]
#               ]
#c2 = create_bubble( c2_centres, rad2,x, interface_width)


p0 = max_values([a1,a2,a3])
p1 = max_values([b1,b2,b3,b4])
p2 = max_values([c1,c2,c3])

p_init = as_vector([p0,p1,p2])


U.sub(1).interpolate(p_init)

# Since using a quadratic potential, we can just get initial values from expansion point
pt = pot.additional_fields['expansion_point']
ci = as_matrix([
    [pt['c0_a']/pt['V_a'],0],
    [pt['c0_b']/pt['V_b'],0],
    [pt['c0_c']/pt['V_c'],0],
    [50*pt['c0_d']/pt['V_d'], pt['c1_d']/pt['V_d']]])
U.sub(0).interpolate(dot(ps,ci)/c_scale)

c_Ar0 = as_vector(ci[-1]) + .05*as_vector([1,-1])

# print("#########")
# print("c_ar = ",c_Ar0)
# print("#########")
# print("c_i1 = ",ci[-1])
# print("#########")

# Boundary conditions
bcs = [
    DirichletBC(V.sub(1).sub(0), Constant(0), 6),
    DirichletBC(V.sub(0),c_Ar0,6),
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
          'ksp_type':'fgmres', 'pc_type':'mg', 'mg_coarse_pc_type':'lu','mg_coarse_pc_factor_mat_solver_type':'mumps', 'ksp_monitor':None,

          }

scheme = time_stepping_scheme(U, test_U, [F_diffusion, F_phase], [],
    time_coefficients = as_vector([1,1,1,1,1]),
    bcs = bcs,
    params = params)

p_n = as_vector([-1,3,1,0])
ps = inner(p_n,ps)

field_names = ['c', 'ps', 'P', 'mu']
writer = writer(['cmesh', 'phase'], field_names, [eval(f) for f in field_names], mesh, "output/output_1e-6D/output.pvd")

solve_time_series(scheme, writer,
    t_range = [0, 1e-6, 1e4],
    iter_t_max = 500,
    eps_t_target = .1,
    eps_s_target = .2)
