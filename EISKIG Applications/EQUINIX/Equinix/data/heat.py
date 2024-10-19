import numpy as np 
import matplotlib.pyplot as plt

dt = 0.15 # time step
# define the three frequencies in radians per sample
omegaT1 = 10000000*np.pi*.73*dt


x=np.arange(1,5,0.001)
y=list()
phi = 0; # phase accumulator
for i in range(0,len(x)):
    c = np.cos(phi) # cosine of current phase
    y.append(c)
   
    phi = phi + omegaT1
  

plt.plot(x, y)
plt.show()