from firedrake import *

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

class time_stepper:
    def __init__(self, time_stepping_scheme, t0 = 0, eps_tol_t = .1):
        self.iteration_time = 0
        self.t0 = t0
        self.eps_tolerance_t= eps_tol_t
        eps_tol_t_target = eps_tol_t/2






class time_stepping_scheme:
    # Defines an object to contain the time stepping
    def __init__(self, U, test_U, F_td, F_qs, tm, bcs=[], dt = 1., nullspace=None, bounds = None, params=[]):

        V = U.function_space()

        self.U = U
        self.U_old = U.copy(deepcopy=True)

        self.dUdt = Function(V)
        self.dUdt_old = Function(V)

        self.dU = TrialFunction(V)

        self.iter = 0
        self.bounds = bounds
        self.dt = Constant(dt)

        F_steady_state = -sum(F_td)+sum(F_qs)
        self.problem_steady_state = NonlinearVariationalProblem(F_steady_state, U, bcs=bcs)
        self.solver_steady_state = NonlinearVariationalSolver(self.problem_steady_state, solver_parameters=params, nullspace=nullspace)

        F = inner(elem_mult(tm,(self.U-self.U_old))/self.dt, test_U)*dx + F_steady_state
        #F = inner(elem_mult(tm,(self.U-self.U_old))/self.dt, as_vector([test_U[5], test_U[6], test_U[2],0,0,0,0,0,0]))*dx - sum(F_td)+sum(F_qs)   #Useful for testing Diffusion
        #F = inner(elem_mult(tm,(self.U-self.U_old)), test_U)*dx - self.dt*sum(F_td)+sum(F_qs)
        self.problem = NonlinearVariationalProblem(F, U, bcs=bcs)
        self.solver = NonlinearVariationalSolver(self.problem, solver_parameters=params, nullspace=nullspace)

    def step(self, dt):
        self.dt.assign(dt)
        #self.t.assign(self.t + dt)

        self.iter += 1
        so = self.solver.solve(bounds = self.bounds)

        self.dUdt.assign(self.U)
        self.dUdt-=self.U_old
        self.dUdt/=self.dt(0)

        #** Not sure why needed? should be self.dUdt.assign((self.U-self.U_old)/self.dt(0))    #Calculate new dUdt
        #self.dUdt.assign((self.U-self.U_old)/self.dt(0))    #Calculate new dUdt
        #print(errornorm(self.dUdt, self.dUdt_old)/self.dt(0)/2*self.dt(0)**2)
        eps_t = errornorm(self.dUdt, self.dUdt_old)/2*self.dt(0) #/dt*dt^2  #Estimate current rate of change of solution
        #eps_t = norm((self.dUdt-self.dUdt_old)/self.dt(0), norm_type='l2')/2*self.dt(0)**2    #Estimate current rate of change of solution

        #eps_t = (self.dUdt.vector()-self.dUdt_old.vector()).max()/self.dt(0)/2*self.dt(0)**2
        #print('max change',self.dUdt.vector().max()*self.dt(0))
        return [so, eps_t, self.dUdt.vector().max()]

    def accept_step(self):
        # If step was acceptable, prepare for next step
        self.dUdt_old.assign(self.dUdt)    # update old dUdt
        self.U_old.assign(self.U)

    def reset_step(self):   # If the solver dies part way through U will have been changed. Need to reset before next round.
        self.U.assign(self.U_old)
        self.dUdt.assign(self.dUdt_old)    # update old dUdt

    def estimate_error_dt2(self):
        return errornorm(self.dUdt, self.dUdt_old)/2*self.dt(0)

    def estimate_error_max_change(self):
        return self.dUdt.vector().max()

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
