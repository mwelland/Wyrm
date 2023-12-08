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
