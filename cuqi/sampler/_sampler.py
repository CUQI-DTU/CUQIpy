from abc import ABC, abstractmethod
import os
import sys
import numpy as np
import pickle as pkl
import cuqi
from cuqi.samples import Samples
from progressbar import progressbar

class SamplerNew(ABC):
    """Abstract base class for all samplers."""

    def __init__(self, target: cuqi.density.Density, initial_point=None, callback=None):
        self.target = target
        self.callback = callback

        # Choose initial point if not given
        if initial_point is None:
            initial_point = np.ones(self.dim)
        
        self._samples = [initial_point]

    # ------------ Public methods ------------

    def sample(self, Ns, batch_size=0, sample_path='./CUQI_samples/'):
        self.batch_size = batch_size
        if(batch_size>0):
            self.sample_path = sample_path
            if self.sample_path.endswith('/'):
                pass
            else:
                self.sample_path += '/'
            if(os.path.exists(self.sample_path)):
                pass
            else:
                os.makedirs(self.sample_path)
        """ Sample Ns samples from the target density. """
        initial_samples_len = len(self._samples)
        for idx in progressbar( range(Ns) ):
            acc = self.step()
            self._acc.append(acc)
            self._samples.append(self.current_point)
            if( (self.batch_size>0) and (idx%self.batch_size==0) ):
                self.dump_samples()

            #self._call_callback(self.current_point, len(self._samples)-1)
            #self._print_progress(len(self._samples), Ns+initial_samples_len)

        return self._samples

    def warmup(self, Nb, tune_freq=0.1):
        """ Warmup the sampler by sampling Nb samples. """
        initial_samples_len = len(self._samples)
        skip_len = int( tune_freq*Nb )
        for idx in progressbar( range(Nb) ):
            acc = self.step()
            if( (idx+1) % skip_len == 0):
                self.tune(skip_len, int(idx/skip_len))
            self._acc.append(acc)
            self._samples.append(self.current_point)
            #self._call_callback(self.current_point, len(self._samples)-1)
            #self._print_progress(len(self._samples), Nb+initial_samples_len)

    def initial_point(self):
        """Return the initial point of the sampler."""
        return self._samples[0]

    def get_samples(self):
        """Return the samples. The internal data-structure for the samples is dynamic so this creates a copy."""
        return Samples(np.array(self._samples), self.target.geometry)

    # ------------ Public properties ------------
    @property
    def dim(self):
        """ Dimension of the target density. """
        return self.target.dim

    @property
    def geometry(self):
        """ Geometry of the target density. """
        return self.target.geometry

    # ------------ Abstract methods ------------
    
    @abstractmethod
    def step(self):
        """Perform one step of the sampler."""
        pass

    @abstractmethod
    def tune(self):
        """Tune the sampler."""
        pass

    # ------------ Abstract methods ------------
    
    @abstractmethod
    def dump_samples(self):
        """Perform one step of the sampler."""
        pass

    # ------------ Abstract properties ------------

    @abstractmethod
#    @property
    def current_point(self):
        """Return the current point of the sampler."""
        pass

    # ------------ Private methods ------------
    def _print_progress(self, s, Ns):
        """Prints sampling progress"""
        # Print sampling progress every 1% of samples and at the end
        if (s % (max(Ns//100,1))) == 0 or s==Ns:
            msg = f'Sampling {s} / {Ns}'
            sys.stdout.write('\r'+msg)
            if s==Ns:
                sys.stdout.write('\n')

    def _call_callback(self, sample, sample_index):
        """ Calls the callback function. Assumes input is sample and sample index"""
        if self.callback is not None:
            self.callback(sample, sample_index)
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

class ProposalBasedSampler(SamplerNew,ABC):
    def __init__(self, target,  proposal=None, scale=1, x0=None, dim=None, **kwargs):
        #TODO: after fixing None dim
        #if dim is None and hasattr(proposal,'dim'):
        #    dim = proposal.dim
        super().__init__(target, initial_point=x0, **kwargs)

        self.current_point = x0
        self.current_target = self.target.logd(self.current_point)
        self.proposal =proposal
        self.scale = scale

        self._acc = [ 1 ]
        self.num_batch_dumped = 0
        self.batch_size = 0

    def dump_samples(self):
        np.savez( self.sample_path + 'batch_{:04d}.npz'.format( self.num_batch_dumped), samples=np.array(self._samples[-1-self.batch_size:] ), batch_id=self.num_batch_dumped )
        self.num_batch_dumped += 1

    @property 
    def proposal(self):
        return self._proposal 

    @proposal.setter 
    def proposal(self, value):
        self._proposal = value

    def save_checkpoint(self, path):
        state = self.get_state()

        with open(path, 'wb') as handle:
            pkl.dump(state, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path):
        with open(path, 'rb') as handle:
            state = pkl.load(handle)

        self.set_state(state)

    def reset(self):
        self._samples.clear()
        self._acc.clear()

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
            return cuqi.geometry._DefaultGeometry(self.dim)

#class ProposalBasedSampler(SamplerNew,ABC):
#    def __init__(self, target,  proposal=None, scale=1, x0=None, dim=None, **kwargs):
#        #TODO: after fixing None dim
#        #if dim is None and hasattr(proposal,'dim'):
#        #    dim = proposal.dim
#
#        super().__init__(target, initial_point=x0, **kwargs)
#
#        self.current_point = x0
#        self.current_target = self.target.logd(self.current_point)
#        self.proposal =proposal
#        self.scale = scale
#
#        self._acc = [ 1 ]
#        self.num_batch_dumped = 0
#        self.batch_size = 0
#
#    def dump_samples(self):
#        np.savez( self.sample_path + 'batch_{:04d}.npz'.format( self.num_batch_dumped), samples=np.array(self._samples[-1-self.batch_size:] ), batch_id=self.num_batch_dumped )
#        self.num_batch_dumped += 1
#
#    def save_checkpoint(self, path):
#        state = self.get_state()
#
#        with open(path, 'wb') as handle:
#            pkl.dump(state, handle, protocol=pkl.HIGHEST_PROTOCOL)
#
#    def load_checkpoint(self, path):
#        with open(path, 'rb') as handle:
#            state = pkl.load(handle)
#
#        print(state)
#
#    def reset(self):
#        self._samples.clear()
#        self._acc.clear()
#
#    @property 
#    def proposal(self):
#        return self._proposal 
#
#    @proposal.setter 
#    def proposal(self, value):
#        self._proposal = value

#    @property
#    def geometry(self):
#        geom1, geom2 = None, None
#        if hasattr(self, 'proposal') and hasattr(self.proposal, 'geometry') and self.proposal.geometry.par_dim is not None:
#            geom1=  self.proposal.geometry
#        if hasattr(self, 'target') and hasattr(self.target, 'geometry') and self.target.geometry.par_dim is not None:
#            geom2 = self.target.geometry
#        if not isinstance(geom1,cuqi.geometry._DefaultGeometry) and geom1 is not None:
#            return geom1
#        elif not isinstance(geom2,cuqi.geometry._DefaultGeometry) and geom2 is not None: 
#            return geom2
#        else:
#            return cuqi.geometry._DefaultGeometry()#(self.dim)
#