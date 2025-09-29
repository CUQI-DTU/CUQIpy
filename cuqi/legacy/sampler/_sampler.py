from abc import ABC, abstractmethod
import sys
import numpy as np
import cuqi
from cuqi.samples import Samples

class Sampler(ABC):

    def __init__(self, target, x0=None, dim=None, callback=None):

        self._dim = dim
        if hasattr(target,'dim'): 
            if self._dim is None:
                self._dim = target.dim 
            elif self._dim != target.dim:
                raise ValueError("'dim' need to be None or equal to 'target.dim'") 
        elif x0 is not None:
            self._dim = len(x0)

        self.target = target

        if x0 is None:
            x0 = np.ones(self.dim)
        self.x0 = x0

        self.callback = callback

    def step(self, x):
        """
        Perform a single MCMC step
        """
        # Currently a hack to get step method for any sampler
        self.x0 = x
        return self.sample(2).samples[:,-1]

    def step_tune(self, x, *args, **kwargs):
        """
        Perform a single MCMC step and tune the sampler. This is used during burn-in.
        """
        # Currently a hack to get step method for any sampler
        out = self.step(x)
        self.tune(*args, *kwargs)
        return out

    def tune(self):
        """
        Tune the sampler parameters.
        """
        pass


    @property
    def geometry(self):
        if hasattr(self, 'target') and hasattr(self.target, 'geometry'):
            geom =  self.target.geometry
        else:
            geom = cuqi.geometry._DefaultGeometry1D(self.dim)
        return geom

    @property 
    def target(self):
        return self._target 

    @target.setter 
    def target(self, value):
        if  not isinstance(value, cuqi.distribution.Distribution) and callable(value):
            # obtain self.dim
            if self.dim is not None:
                dim = self.dim
            else:
                raise ValueError(f"If 'target' is a lambda function, the parameter 'dim' need to be specified when initializing {self.__class__}.")

            # set target
            self._target = cuqi.distribution.UserDefinedDistribution(logpdf_func=value, dim = dim)

        elif isinstance(value, cuqi.distribution.Distribution):
            self._target = value
        else:
            raise ValueError("'target' need to be either a lambda function or of type 'cuqi.distribution.Distribution'")


    @property
    def dim(self):
        if hasattr(self,'target') and hasattr(self.target,'dim'):
            self._dim = self.target.dim 
        return self._dim
    

    def sample(self,N,Nb=0):
        # Get samples from the samplers sample method
        result = self._sample(N,Nb)
        return self._create_Sample_object(result,N+Nb)

    def sample_adapt(self,N,Nb=0):
        # Get samples from the samplers sample method
        result = self._sample_adapt(N,Nb)
        return self._create_Sample_object(result,N+Nb)

    def _create_Sample_object(self,result,N):
        loglike_eval = None
        acc_rate = None
        if isinstance(result,tuple):
            #Unpack samples+loglike+acc_rate
            s = result[0]
            if len(result)>1: loglike_eval = result[1]
            if len(result)>2: acc_rate = result[2]
            if len(result)>3: raise TypeError("Expected tuple of at most 3 elements from sampling method.")
        else:
            s = result
                
        #Store samples in cuqi samples object if more than 1 sample
        if N==1:
            if len(s) == 1 and isinstance(s,np.ndarray): #Extract single value from numpy array
                s = s.ravel()[0]
            else:
                s = s.flatten()
        else:
            s = Samples(s, self.geometry)#, geometry = self.geometry)
            s.loglike_eval = loglike_eval
            s.acc_rate = acc_rate
        return s

    @abstractmethod
    def _sample(self,N,Nb):
        pass

    @abstractmethod
    def _sample_adapt(self,N,Nb):
        pass

    def _print_progress(self,s,Ns):
        """Prints sampling progress"""
        if Ns > 2:
            if (s % (max(Ns//100,1))) == 0:
                msg = f'Sample {s} / {Ns}'
                sys.stdout.write('\r'+msg)
            if s==Ns:
                msg = f'Sample {s} / {Ns}'
                sys.stdout.write('\r'+msg+'\n')

    def _call_callback(self, sample, sample_index):
        """ Calls the callback function. Assumes input is sample and sample index"""
        if self.callback is not None:
            self.callback(sample, sample_index)

class ProposalBasedSampler(Sampler,ABC):
    def __init__(self, target,  proposal=None, scale=1, x0=None, dim=None, **kwargs):
        #TODO: after fixing None dim
        #if dim is None and hasattr(proposal,'dim'):
        #    dim = proposal.dim
        super().__init__(target, x0=x0, dim=dim, **kwargs)

        self.proposal =proposal
        self.scale = scale


    @property 
    def proposal(self):
        return self._proposal 

    @proposal.setter 
    def proposal(self, value):
        self._proposal = value

    @property
    def geometry(self):
        geom1, geom2 = None, None
        if hasattr(self, 'proposal') and hasattr(self.proposal, 'geometry') and self.proposal.geometry.par_dim is not None:
            geom1=  self.proposal.geometry
        if hasattr(self, 'target') and hasattr(self.target, 'geometry') and self.target.geometry.par_dim is not None:
            geom2 = self.target.geometry
        if not isinstance(geom1,cuqi.geometry._DefaultGeometry) and geom1 is not None:
            return geom1
        elif not isinstance(geom2,cuqi.geometry._DefaultGeometry) and geom2 is not None: 
            return geom2
        else:
            return cuqi.geometry._DefaultGeometry1D(self.dim)
