import thermo_potentials as tp
from thermo_potentials.collect import equil_partition, quadratic
from thermo_potentials.materials.ideal_sympy_components import ideal

def build_potential():
    fa = ideal(2,[2500,0], kappa = 1000)
    fb = ideal(2,[0,2500], kappa = 1000)
    fi = interface_energy0/interface_width*V
    mat_i = material(fi, 'i')

    pot_a = tp.sym_potential(fa, 'a')
    pot_b = tp.sym_potential(fb, 'b')

    pot_comp = tp.sym_composite_potential([pot_a, pot_b], rename=False)

    x = .5
    _, y0, _, _ = equil_partition.equil_partition(pot_comp,['c0', 'c1', 'V'],[1-x,x,1])
    #print('Calculated equilibrium partitioning is ', [(v,y) for v,y in zip(pot_comp.vars, y0)])

    pot_quad = quadratic.quadratic_collection(pot_comp, ['c0', 'c1', 'V_a', 'V_b'], y0 = y0)

    return pot_quad
