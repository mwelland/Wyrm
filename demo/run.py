import os
import numpy as np
from timeit import default_timer as timer

nthread = np.array([8])
Lx = np.array([4])

interface_width = 0.2

#n - number of species, m = number of phases
n = 2
m = 2



#out_file = open("walltime.txt", "w")
#out_file.write("nspecies mphases Lx nthreads walltime interfaceWidth" )



for L in Lx:
    for thread in nthread:
 #       start = timer()
        os.system("mpiexec -n %i python3 ideal_2phase_binary_gm.py %i %i %d %f" % (thread,n,m,L,interface_width))
  #      end = timer()
  #      time = end-start

       
   #     out_file.write("%i %i %d %i %f %f" % (n,m,L,thread,time,interface_width))


#out_file.close()
