from abc import ABC, abstractmethod
import os
import sys
import numpy as np
import pickle as pkl
import cuqi
from cuqi.samples import Samples
from progressbar import progressbar

class SamplerNew(ABC):
    """ Abstract base class for all samplers.
    
    Provides a common interface for all samplers. The interface includes methods for sampling, warmup and getting the samples in an object oriented way.

    Samples are stored in a list to allow for dynamic growth of the sample set. Returning samples is done by creating a new Samples object from the list of samples.

    """

    def __init__(self, target: cuqi.density.Density, initial_point=None, callback=None):
        """ Initializer for abstract base class for all samplers.
        
        Parameters
        ----------
        target : cuqi.density.Density
            The target density.

        initial_point : array-like, optional
            The initial point for the sampler. If not given, the sampler will choose an initial point.

        callback : callable, optional
            A function that will be called after each sample is drawn. The function should take two arguments: the sample and the index of the sample.
            The sample is a 1D numpy array and the index is an integer. The callback function is useful for monitoring the sampler during sampling.

        """

        self.target = target
        self.callback = callback

        # Choose initial point if not given
        if initial_point is None:
            initial_point = np.ones(self.dim)
        
        self._samples = [initial_point]

    # ------------ Abstract methods to be implemented by subclasses ------------
    
    @abstractmethod
    def step(self):
        """ Perform one step of the sampler by transitioning the current point to a new point according to the sampler's transition kernel. """
        pass

    @abstractmethod
    def tune(self):
        """ Tune the parameters of the sampler. This method is called after each step of the warmup phase. """
        pass

    @abstractmethod
    def validate_target(self):
        """ Validate the target is compatible with the sampler. Called when the target is set. Should raise an error if the target is not compatible. """
        pass


    # ------------ Public attributes ------------

    @property
    def initial_point(self):
        """ Return the initial point of the sampler. This is always the first sample. """
        return self._samples[0]
    
    @property
    def dim(self):
        """ Dimension of the target density. """
        return self.target.dim

    @property
    def geometry(self):
        """ Geometry of the target density. """
        return self.target.geometry
    
    @property
    def target(self) -> cuqi.density.Density:
        """ Return the target density. """
        return self._target
    
    @target.setter
    def target(self, value):
        """ Set the target density. Runs validation of the target. """
        self._target = value
        self.validate_target()


    # ------------ Public methods ------------

    def get_samples(self) -> Samples:
        """ Return the samples. The internal data-structure for the samples is a dynamic list so this creates a copy. """
        return Samples(np.array(self._samples), self.target.geometry)

    def sample(self, Ns, batch_size=0, sample_path='./CUQI_samples/') -> 'SamplerNew':
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
                
        return self

    def warmup(self, Nb, tune_freq=0.1) -> 'SamplerNew':
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

        return self

    def _call_callback(self, sample, sample_index):
        """ Calls the callback function. Assumes input is sample and sample index"""
        if self.callback is not None:
            self.callback(sample, sample_index)

    # ------------ The following could be moved to base class? ------------

    def dump_samples(self):
        np.savez( self.sample_path + 'batch_{:04d}.npz'.format( self.num_batch_dumped), samples=np.array(self._samples[-1-self.batch_size:] ), batch_id=self.num_batch_dumped )
        self.num_batch_dumped += 1

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

class ProposalBasedSamplerNew(SamplerNew,ABC):
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
            return cuqi.geometry._DefaultGeometry(self.dim)
