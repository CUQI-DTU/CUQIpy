#%%
import sys

sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt

n = 128

X = cuqi.distribution.Cauchy_diff(np.zeros(n), 0.01, physical_dim=1, bc_type="neumann")

# %%
samples = cuqi.sampler.NUTS(X).sample(100, 100)

# %%
samples.plot()

# %%
# The hack to sample Laplace_diff
import scipy as sp

# Least squares form
class Laplace_diff_sampler:

    def __init__(self, prior, beta=1e-6):
        self.prior = prior
        self.beta = beta
        self.x0 = np.ones(prior.dim)

    def sample(self, Ns, Nb=0):

        # Extract diff_op from target prior
        D = self.prior._diff_op
        n = D.shape[0]

        # Gaussian approximation of Laplace_diff prior as function of x_k
        def Lk_fun(x_k):
            dd =  1/np.sqrt((D @ x_k)**2 + self.beta*np.ones(n))
            W = sp.sparse.diags(dd)
            return W.sqrt() @ D

        self.L = Lk_fun(self.x0)
        self.Lmu = self.L@self.prior.location

        def M(x, flag):
            if flag == 1:
                out = np.sqrt(1/self.prior.scale)*(self.L @ x)
            elif flag == 2:
                out = np.sqrt(1/self.prior.scale)*(self.L.T @ x)           
            return out 

        # Initialize samples
        N = Ns+Nb   # number of simulations        
        samples = np.empty((self.prior.dim, N))
                     
        # initial state   
        samples[:, 0] = self.x0
        for s in range(N-1):

            # Update Laplace approximation
            self.L = Lk_fun(samples[:, s])
            self.Lmu = self.L@self.prior.location
        
            # Sample from approximate posterior
            e = cuqi.distribution.Normal(mean=np.zeros(len(self.Lmu)), std=1).sample()
            y = self.Lmu + e # Perturb data
            sim = cuqi.solver.CGLS(M, y, samples[:, s], 1000, 1e-6, 0)            
            samples[:, s+1], _ = sim.solve()

            self._print_progress(s+2,N) #s+2 is the sample number, s+1 is index assuming x0 is the first sample

        # remove burn-in
        samples = samples[:, Nb:]

        return cuqi.samples.Samples(samples, geometry=self.prior.geometry)

    def _print_progress(self,s,Ns):
        """Prints sampling progress"""
        if (s % (max(Ns//100,1))) == 0:
            msg = f'Sample {s} / {Ns}'
            sys.stdout.write('\r'+msg)
        if s==Ns:
            msg = f'Sample {s} / {Ns}'
            sys.stdout.write('\r'+msg+'\n')


# %%
Y = cuqi.distribution.Laplace_diff(np.zeros(n), 0.1, bc_type="neumann", physical_dim=1)
sampler = Laplace_diff_sampler(Y)
samples = sampler.sample(100, 100)
# %%
samples.plot() # Plot the last sample

# %%
# Hack using the sampler in CUQIpy
model = cuqi.model.LinearModel(lambda x: 0*x, lambda y: 0*y, n, n)
likelihood = cuqi.distribution.GaussianCov(model, 1).to_likelihood(np.zeros(n)) # p(y|x)=constant
P = cuqi.distribution.Posterior(likelihood, Y)
samples2 = cuqi.sampler.UnadjustedLaplaceApproximation(P, beta=1e-6, maxit=1000, tol=1e-6, x0=np.ones(n)).sample(100, 100)
# %%
samples2.plot() # Plot the last sample

# %%

# Laplace random walk:
Z = cuqi.distribution.Laplace(np.zeros(n), 10)
d = Z.sample()
x = np.zeros(n)
for i in range(n-1):
    x[i+1] = x[i]+d[i]

plt.plot(x)
# %%
# Cauchy random walk:
d = sp.stats.cauchy.rvs(scale=0.1, size=n)
x = np.zeros(n)
for i in range(n-1):
    x[i+1] = x[i]+d[i]
plt.plot(x)
# %%
