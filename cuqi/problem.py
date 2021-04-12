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
        if isinstance(self.model, cuqi.model.LinearModel) and isinstance(self.likelihood,cuqi.distribution.Gaussian) and isinstance(self.prior,cuqi.distribution.Gaussian):
            
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
        
        #If no implementation exists give error
        else:
            raise NotImplementedError(f'MAP estimate is not implemented in Type1 problem for model: {type(self.model)}, likelihood: {type(self.likelihood)} and prior: {type(self.prior)}. Check documentation for available combinations.')


        
    def sample(self,Ns=100):

        # Gaussian Likelihood, Cauchy prior
        if isinstance(self.likelihood, cuqi.distribution.Gaussian) and isinstance(self.prior, cuqi.distribution.Cauchy_diff):
            
            # Dimension
            n = self.prior.dim
            
            # Set up target and proposal
            def target(x): return self.likelihood.logpdf(self.data,x) + self.prior.logpdf(x) #ToDo: Likelihood should only depend on x (not data)
            def proposal(x_t, sigma): return np.random.normal(x_t, sigma)

            # Set up sampler
            scale = 0.05*np.ones(n)
            x0 = 0.5*np.ones(n)
            MCMC = cuqi.sampler.CWMH(target, proposal, scale, x0)
            
            # Run sampler
            Nb = int(0.2*Ns)   # burn-in
            ti = time.time()
            x_s, target_eval, acc = MCMC.sample_adapt(Ns,Nb); #ToDo: Make results class
            print('Elapsed time:', time.time() - ti)
            
            return x_s
        
        # Gaussian Likelihood, Gaussian prior, linear model (closed-form expression)
        elif isinstance(self.likelihood, cuqi.distribution.Gaussian) and isinstance(self.prior, cuqi.distribution.Gaussian) and not isinstance(self.prior, cuqi.distribution.GMRF) and isinstance(self.model, cuqi.model.LinearModel): 
            
            # Start timing
            ti = time.time()

            A  = self.model.get_matrix()
            b  = self.data
            Ce = self.likelihood.Sigma
            x0 = self.prior.mean
            Cx = self.prior.Sigma

            # Preallocate samples
            n = self.prior.dim 
            x_s = np.zeros((n,Ns))

            x_map = self.MAP() #Compute MAP estimate
            C = np.linalg.inv(A.T@(np.linalg.inv(Ce)@A)+np.linalg.inv(Cx))
            L = np.linalg.cholesky(C)
            for s in range(Ns):
                x_s[:,s] = x_map + L@np.random.randn(n)
                # display iterations 
                if (s % 5e2) == 0:
                    print("\r",'Sample', s, '/', Ns, end="")

            print("\r",'Sample', s+1, '/', Ns)
            print('Elapsed time:', time.time() - ti)
            
            return x_s

        # Gaussian Likelihood, Gaussian prior
        elif isinstance(self.likelihood, cuqi.distribution.Gaussian) and isinstance(self.prior, cuqi.distribution.Gaussian):
            
            # Dimension
            n = self.prior.dim
            
            # Set up target and proposal
            def target(x): return self.likelihood.logpdf(self.data,x) #ToDo: Likelihood should only depend on x (not data)
            #def proposal(ns): return self.prior.sample(ns)
            
            scale = 0.02
            x0 = np.zeros(n)
            
            #ToDO: Switch to pCN
            MCMC = cuqi.sampler.pCN(self.prior,target,scale,x0)
            
            #Run sampler
            ti = time.time()
            x_s, target_eval, acc = MCMC.sample(Ns,0) #ToDo: fix sampler input
            print('Elapsed time:', time.time() - ti)
            
            return x_s
            
        #If no implementation exists give error
        else:
            raise NotImplementedError(f'Sampler is not implemented in Type1 problem for model: {type(self.model)}, likelihood: {type(self.likelihood)} and prior: {type(self.prior)}. Check documentation for available combinations.')
        

            
class Type2(object):
    def __init__(self):
        raise NotImplementedError
