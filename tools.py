from firedrake import *


import time
class timer():
    def __init__(self):
        self.t = time.perf_counter()

    def ping(self, str=None):
        t = time.perf_counter()
        if str:
            print(str, t-self.t)
        else:
            print('Time:', t-self.t)
        self.t = t

def voigt(u):
    return as_vector([u[i,i] for i in range(3)]
                     + [2*u[1,2]]
                     + [2*u[0,2]]
                     + [2*u[0,1]])

def unvoigt(u):
    return as_matrix([[u[0],.5*u[5],.5*u[4]], [.5*u[5], u[1], .5*u[3]],[.5*u[4], .5*u[3], u[2]]])

def epsd2epsd5(eps):
    return

def flc1hs(x,x0,w=[]):
    if not w:
        w = x0
    xp = (x-x0)/w
    smooth = xp**3*(6*xp**2-15*xp+10)
    return conditional(lt(xp,0), 0, conditional(gt(xp,1),1, smooth) )

def rigid_body_np(V_disp):
    x= SpatialCoordinate(V_disp.mesh())
    ns_full = [
        Constant((1, 0, 0)),
        Constant((0, 1, 0)),
        Constant((0, 0, 1)),
        as_vector([-x[1], x[0],     0]),
        as_vector([-x[2],    0,  x[0]]),
        as_vector([    0, -x[2], x[1]]),
        ]
    nsb_full = [interpolate(n, V_disp) for n in ns_full]
    ns_disp = VectorSpaceBasis(nsb_full)
    ns_disp.orthonormalize()
    return ns_disp


class writer:
    def __init__(self, names, field_names, fields, mesh):
        #Names of native fields and user-defined must be treated differently.
        self.names = names  #Names of natrive fields
        self.field_names = field_names  #Names of user-defined fields
        self.fields = fields    #User-defined fields
        self.mesh = mesh

        self.file = File("output/output.pvd")

        # Demanding vectors of length 3. TODO: REvamp. This excludes arrays in native variables. Paraview okay with Arrays (not labelled vectors. Where issue? Firedrake?
        for i in range(len(fields)):
            f = fields[i]
            if len(f.ufl_shape) ==1:
                if f.ufl_shape[0]>3:
                    print('Warning: Vectors are truncated to 3 elements')
                    fields[i] = as_vector([f[0], f[1], f[2]])

        def funcSpace(field, mesh):
            #print(field)
            fd = len(field.ufl_shape)
            if fd==0:
                space = FunctionSpace(mesh, "CG", 1)
            elif fd ==1:
                space = VectorFunctionSpace(mesh, "CG", 1, dim=field.ufl_shape[0])
            elif fd ==2:
                space = TensorFunctionSpace(mesh, "CG", 1, symmetry=True)
            return space
        self.interps = [Interpolator(field, funcSpace(field, mesh)) for field in fields] #List of interpolator objects
        #self.fcns = [i.interpolate().rename(name, name) for i,name in zip(self.interps, self.field_names)]   # list of functions to hold interpolants.
        self.fcns = [i.interpolate() for i in self.interps]   # list of functions to hold interpolants.
        [f.rename(name, name) for f, name in zip(self.fcns, self.field_names)]

        #f = Interpolator.interpolate().rename(field_name, field_name)


        # self.file = XDMFFile(MPI.comm_world, 'solution.xdmf')
        # self.file.parameters['flush_output']=True
        # self.file.parameters['rewrite_function_mesh']=False
        # self.file.parameters["functions_share_mesh"] = True

    def write(self,U,time):
        def get_functions(U,names): # Returns tuple of function with correct names
            sol = U.split()
            for i in range(len(sol)):
               sol[i].rename(names[i],names[i])
            return sol

        #Find function space of field expression (dim?)
        #Define function and interpolator
        # Every timestep, updated functions via interpoaltor (stored for efficiency)
        #print list of functions at each iteration.

        #print('Writing solution')
        flds = list(get_functions(U,self.names))

        [i.interpolate(output=f) for i,f in zip(self.interps, self.fcns)]  #updates all output functions

        #print(flds)
        #flds = flds[:-1]
        #flds = flds + [getFcn(self.fields[i], self.field_names[i],self.mesh) for i in range(len(self.fields))]
        flds = flds + self.fcns

        self.file.write(*flds, time=time)
        #[self.file.write(s,time=time) for s in get_functions(U,self.names)]
        #[self.file.write(getFcn(self.fields[i], self.field_names[i], self.mesh), time) for i in range(len(self.fields))]

class timestepper:
    def __init__(self, U, test_U, F_td, F_qs, tm, bcs=[], dt = 1., nullspace=None, bounds = None, params=[]):
        self.U = U
        self.U_old = U.copy(deepcopy=True)
        V = U.function_space()
        self.dUdt = Function(V)
        self.dUdt_old = Function(V)
        self.dU = TrialFunction(V)
        self.iter = 0
        self.bounds = bounds
        self.dt = Constant(dt)

        F_ss = -sum(F_td)+sum(F_qs)
        self.problem_ss = NonlinearVariationalProblem(F_ss, U, bcs=bcs)
        self.solver_ss = NonlinearVariationalSolver(self.problem_ss, solver_parameters=params, nullspace=nullspace)

        F = inner(elem_mult(tm,(self.U-self.U_old))/self.dt, test_U)*dx +F_ss

        #F = inner(elem_mult(tm,(self.U-self.U_old))/self.dt, as_vector([test_U[5], test_U[6], test_U[2],0,0,0,0,0,0]))*dx - sum(F_td)+sum(F_qs)   #Useful for testing Diffusion


        #F = inner(elem_mult(tm,(self.U-self.U_old)), test_U)*dx - self.dt*sum(F_td)+sum(F_qs)
        self.problem = NonlinearVariationalProblem(F, U, bcs=bcs)
        self.solver = NonlinearVariationalSolver(self.problem, solver_parameters=params, nullspace=nullspace)

    def step(self, dt):
        self.dt.assign(dt)
        self.iter += 1
        so = self.solver.solve(bounds = self.bounds) #If this errors out, nothing else will execute
        # Only reach this point if solve was successful
        self.dUdt.assign(self.U)
        self.dUdt-=self.U_old
        self.dUdt/=self.dt(0)

        #** Not sure why needed? should be self.dUdt.assign((self.U-self.U_old)/self.dt(0))    #Calculate new dUdt
        #self.dUdt.assign((self.U-self.U_old)/self.dt(0))    #Calculate new dUdt
        #print('tick')
        #print(errornorm(self.dUdt, self.dUdt_old)/self.dt(0)/2*self.dt(0)**2)
        eps_t = errornorm(self.dUdt, self.dUdt_old)/2*self.dt(0) #/dt*dt^2  #Estimate current rate of change of solution
        #print('prick')
        #eps_t = norm((self.dUdt-self.dUdt_old)/self.dt(0), norm_type='l2')/2*self.dt(0)**2    #Estimate current rate of change of solution
        #print('tock', eps_t)


        #eps_t = (self.dUdt.vector()-self.dUdt_old.vector()).max()/self.dt(0)/2*self.dt(0)**2
        #print('max change',self.dUdt.vector().max()*self.dt(0))
        return [so, eps_t, self.dUdt.vector().max()]

    def accept_step(self):
        # If step was acceptable, prepare for next step
        self.dUdt_old.assign(self.dUdt)    # update old dUdt
        self.U_old.assign(self.U)

    def reset_step(self):   # If the solver dies part way through U will have been changed. Need to reset before next round.
        self.U.assign(self.U_old)

    def plotJac(self):
        petsc_mat = assemble(self.problem.J).M.handle    # Not sure if needed or already assembled?

        import scipy.sparse as sp
        import matplotlib.pyplot as plt
        indptr, indices, data = petsc_mat.getValuesCSR()
        scipy_mat = sp.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())
        plt.spy(scipy_mat)
        plt.savefig('Jacobian.png')
        #plt.show()

    def steady_state(self):
        so = self.solver_ss.solve() #If this errors out, nothing else will execute
        print("Solved!")
        # Only reach this point if solve was successful
        self.dUdt.assign(self.U)
        self.dUdt-=self.U_old
        deltaU = self.U-self.U_old
        self.dUdt/=self.dt(0)
        #self.dUdt.assign((self.U.vector()-self.U_old.vector())/self.dt(0))    #Calculate new dUdt
        eps_t = norm((self.dUdt-self.dUdt_old)/self.dt(0), norm_type='l2')/2*self.dt(0)**2
        return [so, eps_t, self.deltaU.vector().max()]

    #def take_step(self):



def getGmsh(Lx, Ly, Lz, res):
    import os.path
    if not os.path.isfile('mesh.msh'):
        gmshBoxMesh(Lx, Ly, Lz, res)


def gmshBoxMesh(Lx, Ly, Lz, res):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("out")
    #gmsh.model.occ.addBox(-Lx/2, -Lx/2, -Lx/2, Lx, Lx, Lx )
    gmsh.model.occ.addBox(0, 0, 0, Lx, Lx, Lx )
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), Lx/3)
    #gmsh.model.mesh.setSize([(0, 1)], 0.2)
    gmsh.model.mesh.generate(3)
    gmsh.write("mesh.msh")
    gmsh.finalize()


# import pygmsh
# class msh:
#     def __init__(self, Lx, Ly, Lz, mesh_res_min, mesh_res_max=[]):
#         #geom = pygmsh.geo.Geometry()
#         if not mesh_res_max:
#             mesh_res_max = 10*mesh_res_min
#         self.Lx = Lx
#         self.Ly = Ly
#         self.Lz = Lz
#         self.mesh_res_min = mesh_res_min
#         self.mesh_res_max = mesh_res_max
#         self.mesh_refine_pts = []
#         with pygmsh.geo.Geometry() as geom:
#
#             rectangle = geom.add_rectangle(0.0, self.Lx, 0.0, self.Ly, 0, self.mesh_res_max)
#             mesh = geom.generate_mesh(dim=2)
#             pygmsh.write("test.msh")
#
#         #pygmsh.write("test.msh")
#         #return Mesh("test.msh")
#
#     def refine(self, points):
#         with pygmsh.geo.Geometry() as geom:
#             rectangle = geom.add_rectangle(0.0, self.Lx, 0.0, self.Ly, 0, self.mesh_res_max)
#             pts = [geom.add_point(p) for p in points]
#             [geom.in_surface(p, rectangle) for p in pts]
#
#             field0 = geom.add_boundary_layer(
#                 nodes_list=pts,
#                 lcmin=self.mesh_res_min,
#                 lcmax=self.mesh_res_max,
#                 distmin=self.mesh_res_min,
#                 distmax=self.mesh_res_max,
#             )
#             geom.set_background_mesh([field0], operator="Min")
#             #return self.get_mesh()
#             mesh = geom.generate_mesh(dim=2)
#             pygmsh.write("test.msh")
#         return Mesh("test.msh")
#
#     def get_mesh(self):
#         mesh = self.geom.generate_mesh(dim=2)
#         pygmsh.write("test.msh")
#         return Mesh("test.msh")
#



# def msh(points, Lx, Ly, Lz, mesh_res_min, mesh_res_max=[]):
#
#     with pygmsh.geo.Geometry() as geom:
#         rectangle = geom.add_rectangle(0.0, Lx, 0.0, Ly, 0, mesh_res_max)
#
#
#         pts = [geom.add_point(p) for p in points]
#         [geom.in_surface(p, rectangle) for p in pts]
#
#         field0 = geom.add_boundary_layer(
#             nodes_list=pts,
#             lcmin=mesh_res_min,
#             lcmax=mesh_res_max,
#             distmin=mesh_res_min,
#             distmax=mesh_res_max,
#         )
#         geom.set_background_mesh([field0], operator="Min")
#
#         mesh = geom.generate_mesh(dim=2)
#         pygmsh.write("test.msh")
#         return Mesh("test.msh")
#         print('writen')







#
# import pygmsh, copy, gmsh
# class msh:
#     def __init__(self, Lx, Ly, Lz, mesh_coarse = [], mesh_fine = []):
#         if not mesh_coarse:
#             self.mesh_coarse = Lx/10
#         if not mesh_fine:
#             self.mesh_fine = self.mesh_coarse/5
#         self.Lx = Lx
#         self.Ly = Ly
#         self.Lz = Lz
#         #
#         #
#
#         #self.working = copy.deepcopy(self.base)
#
#     def remesh(self, pts):
#         with pygmsh.geo.Geometry() as geom:
#             geom = pygmsh.geo.Geometry()
#
#             rect = geom.add_rectangle(0.0, 1., 0.0, 1., 0)
#             #msh_pts = [geom.add_point(p + [0], self.mesh_fine) for p in pts]
#             #pt1 = geom.add_point([.2,.2,0],.1)
#
#             field1 = geom.add_boundary_layer(
#                 nodes_list=[geom.add_point([.2,.2,0],.1)],
#                 lcmin=0.01,
#                 lcmax=0.1,
#                 distmin=0.0,
#                 distmax=0.2,
#                 )
#
#
#             geom.set_background_mesh([field1], operator='Min')
#             #[geom.in_surface(msh_pt, rect) for msh_pt in msh_pts]
#             #geom.set_recombined_surfaces([rect.surface])
#             #geom.set_background_mesh([field0, field1], operator="Min")
#
#             #gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
#             #opt_mesh = pygmsh.optimize(mesh,method="")
#             mesh = geom.generate_mesh(dim=2)
#             mesh.write("test.vtk")
#
#             pygmsh.write("test.msh")
#
#         print('mesh written')

    # def remesh(self, pts, mesh_res_fine):
    #     self.working = copy.deepcopy(self.base)
    #     p1 = self.working.add_point([.5, .5, 0.], mesh_res)
    #     self.working.in_surface(p1, se)
