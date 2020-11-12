import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt

from cuqi.ProbDist.Distribution import Gaussian

# 2D Gaussian
d = 2
mean = np.array([0, 0])
std = np.array([1, 1])
R = np.array([[1,-0.7],[-0.7,1]])
pX_1 = Gaussian(mean, std, R)

# draw samples
N = 10
s = pX_1.sample(N)

# evaluation of PDF
m, n = 200, 200
X, Y = np.meshgrid(np.linspace(-2, 2.5, m), np.linspace(-2, 2.5, n))
Xf, Yf = X.flatten(), Y.flatten()
pos = np.vstack([Xf, Yf]).T   # pos is (m*n, d)
Z = pX_1.pdf(pos, []).reshape((m, n))

# plot
plt.figure()
plt.contourf(X, Y, Z, 20)
plt.contour(X, Y, Z, 20, colors='k')
# plt.plot(s[:,0], s[:,1], 'r*')
plt.gca().set_aspect('equal', adjustable='box')
plt.pause(5)