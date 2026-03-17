"""This function simulates the projection of a line segement defined by 2 points (-5, Z) and
(5, Z), where Z ranges from 10 to 1000, assuming a camera focal length of f = 1. By computing
the following 2 tasks:

1) For each distance Z, the two points are projected into a 1-D sensor under perspective project-
ion, and compute the length of the segement, x.

2) The length is plotted as a function of distance Z to see how size changes as a function of dis-
tance Z to see how size changes as a function of distance to the camera."""

#Import
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#Create Z
Z = np.arange(10, 1001, 10)
print(Z)

#Create X
X_1 = np.ones(len(Z))*5
X_2 = np.ones(len(Z))*(-5)
print(X_1, X_2)
#Create f
f = 1

#Calculate x
x =  -X_2*f/Z - (-X_1*f/Z)
print(x)

#Plot x against Z
plt.plot(Z, x, 'bo--')
plt.ylabel("Image segement length, x")
plt.xlabel("Distance real segement from aperture, Z")
plt.show()

#Alternative solution
X1 = 5
X2 = -5
f = 1

L = []
for Z in range(10, 1001):
    x1 = -f*X1/Z
    x2 = -f*X2/Z
    L.append(sqrt((x1-x2)**2))

plt.plot(L)
plt.ylabel("Image segement length, x")
plt.xlabel("Distance real segement from aperture, Z")
plt.show()