#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
#from mcmc import Random_Walk
import sys
sys.path.append("../..") 
import cuqi
from heatmodel import heat

#%% computing the true initial condition
# alpha = x * exp(-2x) * sin(pi-x)
N = 128
dx = np.pi/(N+1)
x = np.linspace(dx,np.pi,N,endpoint=False)
true_init = x*np.exp(-2*x)*np.sin(np.pi-x)
plt.plot(true_init)

#%% defining the heat equation as the forward map
model = heat(N=N)
model.set_init_cond(true_init)
model.time_stepping()
y_obs = model.advance_with_init_cond(true_init) # observation vector

plt.plot(y_obs)

#%%
SNR = 100 # signal to noise ratio
sigma = np.linalg.norm(y_obs)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise

likelihood = cuqi.distribution.Gaussian(model,sigma,np.eye(N))

#%% Prior
prior = cuqi.distribution.Gaussian(np.zeros((N,)),1)

target = lambda xx: likelihood(x=y_obs).logpdf(xx) + prior.logpdf(xx)
proposal = cuqi.distribution.Gaussian(np.zeros((N,)),1)
scale = 1.0
init_x = np.zeros((N))

mysampler = cuqi.sampler.RWMH(proposal, target, scale, init_x)


#IP=cuqi.problem.BayesianProblem(likelihood,prior,y_obs)

results=mysampler.sample_adapt(100,20)

#%%
# formulating the Bayesian inverse problem
pi_like = lambda p: - ( 0.5*np.sum(  (prob.forward(p) - y_obs)**2)/sigma2) # least squares likelihood
p0 = np.zeros(N) # initial parameter guess
Ns = 10000 # number of samples
Nb = 10 # number of burn-in samples
RWM = Random_Walk(pi_like,p0) # type of MCMC (Metropolis)
RWM.sample(Ns,Nb) # running MCMC

RWM.print_stat() # this prints the acceptance rate
samples, target = RWM.give_stats() # extracting samples

# computing the mean parameter
p_mean = np.mean(samples[300:,:],axis=0)

# computing the mean field
field = rand_field(N)
field.set_params(p_mean)
f = field.give_real_func()

# plotting the mean solution vs the true solution
fig, ax1 = plt.subplots(1)
ax1.plot(x,f,label=r'posterior mean',color="blue",linewidth=2.)
ax1.plot(x,true_init,label=r'true initial condition',color="orange",linewidth=2.)
plt.show()



'''

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



from heatmodel import heat

HM = heat()

#%%
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
'''