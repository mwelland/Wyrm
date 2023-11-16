import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


df_mg = pd.read_csv("walltime.txt",sep=' ',header=0)
df_direct = pd.read_csv("walltime_direct.txt",sep=' ',header=0)

plt.plot(df_direct['Lx'],df_direct['walltime'],label='Direct')
plt.plot(df_mg['Lx'],df_mg['walltime'],label='MG')
plt.legend()
plt.xlabel("Lx")
plt.ylabel("wall time (s)")
plt.show()