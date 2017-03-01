import numpy as np


mass = np.array([0.0,0.0])

powed = np.power(mass,-1)
print(powed)
powed[powed>1e10]=1e10
print(powed)