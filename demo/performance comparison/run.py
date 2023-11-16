import os
import numpy as np
from timeit import default_timer as timer
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/Wyrm")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/Wyrm/demo")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/thermo_potentials")
#os.system("source /home/george/firedrake/firedrake/bin/activate")
nthread = np.array([8])
Lx = np.array([10])



#n - number of species, m = number of phases
n = 2
m = 2



out_file = open("walltime.txt", "a")
#out_file.write("nspecies mphases Lx nthreads walltime\n" )



for L in Lx:
    for thread in nthread:
        start = timer()
        os.system("mpinexec -n 8 python3 ideal_2phase_binary_gm.py %i %i %d" % (n,m,L))
        #os.system("python3 ideal_2phase_binary.py %i %i %d" % (n,m,L))
        end = timer()
        time = end-start

       
        out_file.write("%i %i %d %i %f\n" % (n,m,L,thread,time))


out_file.close()