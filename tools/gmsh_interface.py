from firedrake import *

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
