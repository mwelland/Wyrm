from firedrake import *
from firedrake.petsc import PETSc
from numpy import random
from firedrake.__future__ import interpolate



def print(*args, **kwargs):
    #Overloads print to be the petsc routine which relegates to the head mpi rank
    PETSc.Sys.Print(*args,flush=True)

class writer:
    def __init__(self, names, field_names, fields, mesh, filename = "output/output.pvd"):
        #Names of native fields and user-defined must be treated differently.
        self.names = names  #Names of calculated fields
        self.field_names = field_names  #Names of user-defined fields
        self.fields = fields    #User-defined fields
        self.mesh = mesh
        self.filename = filename

        self.file = File(self.filename)

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
        #self.interps = [Interpolator(field, funcSpace(field, mesh)) for field in fields] #List of interpolator objects
        self.interps = [interpolate(field, funcSpace(field, mesh)) for field in fields] #List of interpolator objects
        #self.fcns = [i.interpolate().rename(name, name) for i,name in zip(self.interps, self.field_names)]   # list of functions to hold interpolants.
        #self.fcns = [i.interpolate() for i in self.interps]   # list of functions to hold interpolants.
        self.fcns = [assemble(i) for i in self.interps]   # list of functions to hold interpolants.
        [f.rename(name, name) for f, name in zip(self.fcns, self.field_names)]

    def write(self, U, time):
        def get_functions(U, names): # Returns tuple of function with correct names
            sol = U.subfunctions
            for i in range(len(sol)):
                sol[i].rename(names[i],names[i])
            return sol

        #print('Writing solution')
        flds = list(get_functions(U, self.names))

        [assemble(i) for i,f in zip(self.interps, self.fcns)]  #updates all output functions

        #print(flds)
        #flds = flds[:-1]
        #flds = flds + [getFcn(self.fields[i], self.field_names[i],self.mesh) for i in range(len(self.fields))]
        flds = flds + self.fcns

        self.file.write(*flds, time=time)
        #[self.file.write(s,time=time) for s in get_functions(U,self.names)]
        #[self.file.write(getFcn(self.fields[i], self.field_names[i], self.mesh), time) for i in range(len(self.fields))]



def solve_time_series(scheme, writer,
            t_range = [0, 5e-2, 1e4],
            eps_t_target = .1,
            eps_t_limit = None,
            eps_s_target = 1000,
            eps_s_limit = 1000,
            exit_on_error = False,
            iter_t_max = 1000,
            max_dt_change = 2,
            ):

    t, dt, t_end = t_range
    iter_t = 0

    # if not eps_t_limit:
    #     eps_t_limit = 2*eps_t_target

    if not eps_s_limit:
        eps_s_limit = 2*eps_s_target

    writer.write(scheme.U, 0.0)
    while t<t_end and iter_t<iter_t_max:
        iter_t +=1
        proceed = False
        print('\n{:n}: Solving for time: {:6.4g}'.format(iter_t, t+dt))
        try:
            so, eps_t, eps_s = scheme.step(dt)
            print('Converged with dt: {:4.2g}. Estimated error: {:4.2g}, max change {:4.2g}'.format(dt, eps_t, eps_s))
            proceed = True
        except KeyboardInterrupt:
            print ('KeyboardInterrupt exception is caught')
            break
        except Exception as ex:
            print('Failed with dt: {:6.4g} \n'.format(dt), ex)
            if exit_on_error:
                raise

        if iter_t == 1:
            scheme.solver.parameters.pop('snes_view',None) # Unset the snes_viewer so as not to repeat it.

        if proceed:
            if eps_t_limit is not None:
                if eps_t>eps_t_limit:
                    print('Time error limit exceeded')
                    proceed = False

        if proceed and eps_s>eps_s_limit:
            print('Max solution change limit exceeded')
            proceed = False

        if proceed:
                # Time step is successful and acceptable
                t += dt
                scheme.accept_step()
                dt *= min(
                    eps_t_target/(eps_t+1e-10),
                    eps_s_target/(eps_s+1e-10),
                    max_dt_change)
        else:
            dt *=.5
            scheme.reset_step()

        # Adapt the time step to some metric
        #dphase = errornorm(phase,phase_old,'l10')
        #print('max phase change', dphase)
        #dphase_target = .1

        #dt.assign(float(dt)*min( (dphase_target/(dphase)), 20))  # Change timestep to aim for tolerance



        if iter_t %1 ==0:
            writer.write(scheme.U, time=float(t))

class time_stepping_scheme:
    # Defines an object to contain the time stepping
    def __init__(self, U, test_U, F_td, F_qs, time_coefficients, bcs=[], dt = 1, nullspace=None, bounds = None, params=[]):

        V = U.function_space()
        self.U = U
        self.U_old = U.copy(deepcopy=True)
        self.dUdt = Function(V)
        self.dUdt_old = Function(V)
        self.dU = TrialFunction(V)

        self.bounds = bounds
        self.dt = Constant(dt)

        F_steady_state = -sum(F_td)+sum(F_qs)
        self.problem_steady_state = NonlinearVariationalProblem(F_steady_state, U, bcs=bcs)
        self.solver_steady_state = NonlinearVariationalSolver(self.problem_steady_state, solver_parameters=params, nullspace=nullspace)

        F_time_dependant = inner(elem_mult(time_coefficients,(self.U-self.U_old))/self.dt, test_U)*dx + F_steady_state
        self.problem = NonlinearVariationalProblem(F_time_dependant, U, bcs=bcs)
        self.solver = NonlinearVariationalSolver(self.problem, solver_parameters=params, nullspace=nullspace)

    def step(self, dt):
        self.dt.assign(dt)
        #self.t.assign(self.t + dt)
        so = self.solver.solve(bounds = self.bounds)
        self.dUdt.assign((self.U-self.U_old))    #Is this slow compared to itnerpolate or vector manipulation?
        self.dUdt /= dt
        eps_t = errornorm(self.dUdt, self.dUdt_old)/2*dt #/dt*dt^2  #Estimate current rate of change of solution
        #eps_s_max = errornorm(self.U, self.U_old,'l100')   #l100 argument doesn't work. no linf option
        eps_s = self.dUdt.vector().max()*dt
        return [so, eps_t, eps_s]

    def accept_step(self):
        self.dUdt_old.assign(self.dUdt)
        self.U_old.assign(self.U)

    def reset_step(self):
        self.U.assign(self.U_old)
        self.dUdt.assign(self.dUdt_old)    # update old dUdt

    def jump_to_steady_state(self):
        so = self.solver_steady_state.solve() #If this errors out, nothing else will execute
        print("Solved!")
        # Only reach this point if solve was successful
        self.dUdt.assign(self.U)
        self.dUdt-=self.U_old
        deltaU = self.U-self.U_old
        self.dUdt/=self.dt(0)
        #self.dUdt.assign((self.U.vector()-self.U_old.vector())/self.dt(0))    #Calculate new dUdt
        eps_t = norm((self.dUdt-self.dUdt_old)/self.dt(0), norm_type='l2')/2*self.dt(0)**2
        return [so, eps_t, self.deltaU.vector().max()]

    def plotJac(self):
        print('Warning - not tested')
        petsc_mat = assemble(self.problem.J).M.handle    # Not sure if needed or already assembled?

        import scipy.sparse as sp
        import matplotlib.pyplot as plt
        indptr, indices, data = petsc_mat.getValuesCSR()
        scipy_mat = sp.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())
        plt.spy(scipy_mat)
        plt.savefig('Jacobian.png')
        #plt.show()

def create_bubble(centre, r, x, interface_width):
    def create_bbl(centre, radius, x, interface_width):
        #creates single bubble
        centre = as_vector(centre)
        r = sqrt(inner(x-centre, x-centre))
        return .5*(1.-tanh((r-radius)/(2.*interface_width)))

    if type(centre) is list:
        #TODO: If radii is list of same length, zip and make multiples
        p_bubbles = [create_bbl(c, r, x, interface_width) for c in centre]
        return max_values(p_bubbles)
    else:
        return create_bbl(centre, r, x, interface_width)

def max_values(lst):
    if len(lst)<=1:
        return lst[0]
    else:
        return max_value(lst[0], max_values(lst[1:]))

def define_centres_arr(lower_edge,upper_edge,step_size,Lx,dims, rand = False, height = 0):
    nrow = int((((upper_edge - lower_edge)/step_size) + 1)**(dims-1))
    print(nrow)
    arr = np.zeros((nrow,dims))
    arr[:,dims-1] = height
    a = [lower_edge]*(dims-1)
    for i in range(len(arr)):

        for j in range(dims-1):

            arr[i,j] = a[j]*Lx
            if rand == True:
                arr[i,j] = round(random.uniform(lower_edge,upper_edge),1)

        if dims == 3:
            a[1] += step_size
            if a[1] > upper_edge:
                a[1] = 0
                a[0] += step_size
        elif dims == 2:
            a[0] += step_size
        else:
            print("Invalid number of dimensions entered.")
            break
    return arr
