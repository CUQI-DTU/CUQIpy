import sys
import time
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import cuqi

#%% %Cuqi deblur test problem
tp = cuqi.testproblem.Deconvolution2D() #Default values

#%% Unpack problem in b = A*x + e
b = tp.data;    # blurred image
A = tp.model    # matrix-free convolution model
e = tp.noise    # Class with noise

#%% plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.figure()
A.domain_geometry.plot(tp.exactSolution, cmap='Greys_r')
plt.figure()
A.range_geometry.plot(b, cmap='Greys_r')
plt.show()
