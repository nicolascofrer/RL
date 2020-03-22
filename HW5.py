import numpy as np
import itertools as it
atEstabblishment=np.array([[1,1], [1,0], [0,1], [1,1], [0,0], [1,0], [1,1]])
fightOccurred = np.array([0, 1, 0, 0, 0, 1, 0])

print(atEstabblishment.shape)
N_patrons=atEstabblishment.shape[1]

H=[p]
