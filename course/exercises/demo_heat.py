import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
#from mcmc import Random_Walk
import sys
sys.path.append("../..") 
import cuqi

N = 128
dx = np.pi/(N+1)
x = np.linspace(dx,np.pi,N,endpoint=False)
true_init = x*np.exp(-2*x)*np.sin(np.pi-x)
plt.plot(true_init)
plt.xlabel('x')
plt.ylabel('u')

from cuqi.geometry import KLField, StepField

geo = KLField(N)

from cuqi.distribution import Normal

P_coeff = Normal(mean=np.zeros((N,)),std=1)

sam = P_coeff.sample()

geo.to_function(sam)

geo.plot(geo.to_function(sam))

geo_step = StepField(N)

step_coeff = np.array([1,2,3])

geo.plot(geo_step.to_function(step_coeff))

#problem = heat.heat()
#problem.set_init_cond(true_init)

# Create CUQI model
#M = cuqi.model.Model(problem.forward, N, N)

#y = M.forward(true_init)

#plt.plot(y)
plt.figure()
plt.plot(true_init)
plt.figure()
plt.plot(geo.to_function(true_init))