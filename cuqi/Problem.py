import cuqi
import numpy as np
import time

class Generic(object):
    def __init__(self):
        raise NotImplementedError

class Type1(object):
    
    def __init__(self,data,model,noise,prior):
        self.data = data
        self.model = model
        self.noise = noise
        self.prior = prior
        
        #Likelihood is just noise dist w. different mean
        self.likelihood = noise
        self.likelihood.mean = lambda *args: self.model.forward(*args)
        
    def MAP(self):
        #MAP computed the MAP estimate of the posterior
        
        #Linear model with Gaussian likelihood and prior
        if isinstance(self.model, cuqi.Model.Linear) and isinstance(self.likelihood,cuqi.Distribution.Gaussian) and isinstance(self.prior,cuqi.Distribution.Gaussian):
            
            A  = self.model.get_matrix()
            b  = self.data
            Ce = self.likelihood.Sigma
            x0 = self.prior.mean
            Cx = self.prior.Sigma

            #Basic map estimate using closed-form expression
            #Tarantola 2005 (3.37-3.38)
            rhs = b-A@x0
            sysm = A@Cx@A.T+Ce
            
            return x0 + Cx@(A.T@np.linalg.solve(sysm,rhs))
        

        
    def sample(self,Ns=100):
        
        # Gaussian Likelihood, Cauchy prior
        if isinstance(self.likelihood, cuqi.Distribution.Gaussian) and isinstance(self.prior, cuqi.Distribution.Cauchy_diff):
            
            # Dimension
            n = self.prior.dim
            
            # Set up target and proposal
            def target(x): return self.likelihood.logpdf(self.data,x) + self.prior.logpdf(x) #ToDo: Likelihood should only depend on x (not data)
            def proposal(x_t, sigma): return np.random.normal(x_t, sigma)

            # Set up sampler
            scale = 0.05*np.ones(n)
            x0 = 0.5*np.ones(n)
            MCMC = cuqi.Sampler.CWMH(target, proposal, scale, x0)
            
            # Run sampler
            Nb = int(0.2*Ns)   # burn-in
            ti = time.time()
            x_s, target_eval, acc = MCMC.sample_adapt(Ns,Nb); #ToDo: Make results class
            print('Elapsed time:', time.time() - ti)
            
            return x_s

            
class Type2(object):
    def __init__(self):
        raise NotImplementedError