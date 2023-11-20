import os
import numpy as np
from timeit import default_timer as timer
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/Wyrm")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/Wyrm/demo")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/")
os.system("export PYTHONPATH=$PYTHONPATH:/home/george/phaseField/thermo_potentials")
#os.system("source /home/george/firedrake/firedrake/bin/activate")
nthread = np.array([8])
Lx = np.array([6,7,8])



#n - number of species, m = number of phases
nSpecies = np.array([2])
m = 2



#out_file = open("walltime_GM_3D.txt", "a")
#out_file.write("nspecies mphases Lx nthreads walltime\n" )


i=0
for n in nSpecies:
    for L in Lx:
        for thread in nthread:
            out_file = open("walltime_GM_3D.txt", "a")
            start = timer()
            os.system("mpiexec -n 8 python3 ideal_2phase_binary_gm.py %i %i %d > out_GM_n=%im=%iL=%di=%i.txt" % (n,m,L,n,m,L,i))
            #os.system("python3 ideal_2phase_binary.py %i %i %d > out_direct_3d_L=%dn=%im=%ii=%i.txt" % (n,m,L,L,n,m,i))
            end = timer()
            time = end-start

        
            out_file.write("%i %i %d %i %f %i\n" % (n,m,L,thread,time,i))
            out_file.close()
            print("##################3")
            print("i = ",i)
            print("##################3")
            i+=1


#out_file.close()