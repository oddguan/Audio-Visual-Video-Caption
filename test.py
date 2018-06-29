import numpy as np 

f = np.zeros((20, 0))
a = np.zeros((20,95))
b = np.zeros((20, 97))
c = np.zeros((20, 97))
d = np.zeros((20, 97))
e = np.zeros((20, 97))
f = np.concatenate((f,a), axis=1)
print(f.shape)