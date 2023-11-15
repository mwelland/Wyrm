import os
import numpy as np
from timeit import default_timer as timer
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/Wyrm")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/thermo_potentials")
nthread = np.array([8])
Lx = np.array([5,10,20,40,50,60,80])



#n - number of species, m = number of phases
n = 2
m = 2



out_file = open("walltime.txt", "w")
out_file.write("nspecies mphases Lx nthreads walltime\n" )



for L in Lx:
    for thread in nthread:
        start = timer()
        os.system("mpiexec -n %i python3 ideal_2phase_binary_gm.py %i %i %d" % (thread,n,m,L))
        end = timer()
        time = end-start

       
        out_file.write("%i %i %d %i %f\n" % (n,m,L,thread,time))


out_file.close()