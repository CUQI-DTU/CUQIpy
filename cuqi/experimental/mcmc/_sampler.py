from abc import ABC, abstractmethod
import os
import numpy as np
import pickle as pkl
import warnings
import cuqi
from cuqi.samples import Samples

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        warnings.warn("Module mcmc: tqdm not found. Install tqdm to get sampling progress.")
        return iterable

class Sampler(ABC):
    """ Abstract base class for all samplers.
    
    Provides a common interface for all samplers. The interface includes methods for sampling, warmup and getting the samples in an object oriented way.

    Samples are stored in a list to allow for dynamic growth of the sample set. Returning samples is done by creating a new Samples object from the list of samples.

    The sampler maintains sets of state and history keys, which are used for features like checkpointing and resuming sampling.

    The state of the sampler represents all variables that are updated (replaced) in a Markov Monte Carlo step, e.g. the current point of the sampler.

    The history of the sampler represents all variables that are updated (appended) in a Markov Monte Carlo step, e.g. the samples and acceptance rates.

    Subclasses should ensure that any new variables that are updated in a Markov Monte Carlo step are added to the state or history keys.

    Saving and loading checkpoints saves and loads the state of the sampler (not the history).

    Batching samples via the batch_size parameter saves the sampler history to disk in batches of the specified size.

    Any other attribute stored as part of the sampler (e.g. target, initial_point) is not supposed to be updated
    during sampling and should not be part of the state or history.

    """

    _STATE_KEYS = {'current_point'}
    """ Set of keys for the state dictionary. """

    _HISTORY_KEYS = {'_samples', '_acc'}
    """ Set of keys for the history dictionary. """

    def __init__(self, target:cuqi.density.Density=None, initial_point=None, callback=None):
        """ Initializer for abstract base class for all samplers.

        Any subclassing samplers should simply store input parameters as part of the __init__ method. 

        The actual initialization of the sampler should be done in the _initialize method.
        
        Parameters
        ----------
        target : cuqi.density.Density
            The target density.

        initial_point : array-like, optional
            The initial point for the sampler. If not given, the sampler will choose an initial point.

        callback : callable, optional
            A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
            The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.
        """

        self.target = target
        self.initial_point = initial_point
        self.callback = callback
        self._is_initialized = False

    def initialize(self):
        """ Initialize the sampler by setting and allocating the state and history before sampling starts. """

        if self._is_initialized:
            raise ValueError("Sampler is already initialized.")
        
        if self.target is None:
            raise ValueError("Cannot initialize sampler without a target density.")
        
        # Default values
        if self.initial_point is None:
            self.initial_point = self._get_default_initial_point(self.dim)

        # State variables
        self.current_point = self.initial_point

        # History variables
        self._samples = []
        self._acc = [ 1 ] # TODO. Check if we need to put 1 here.

        self._initialize() # Subclass specific initialization

        self._validate_initialization()

        self._is_initialized = True

    # ------------ Abstract methods to be implemented by subclasses ------------
    @abstractmethod
    def step(self):
        """ Perform one step of the sampler by transitioning the current point to a new point according to the sampler's transition kernel. """
        pass

    @abstractmethod
    def tune(self, skip_len, update_count):
        """ Tune the parameters of the sampler. This method is called after each step of the warmup phase.

        Parameters
        ----------
        skip_len : int
            Defines the number of steps in between tuning (i.e. the tuning interval).

        update_count : int
            The number of times tuning has been performed. Can be used for internal bookkeeping.
        
        """
        pass

    @abstractmethod
    def validate_target(self):
        """ Validate the target is compatible with the sampler. Called when the target is set. Should raise an error if the target is not compatible. """
        pass

    @abstractmethod
    def _initialize(self):
        """ Subclass specific sampler initialization. Called during the initialization of the sampler which is done before sampling starts. """
        pass

    # ------------ Public attributes ------------
    @property
    def dim(self) -> int:
        """ Dimension of the target density. """
        return self.target.dim

    @property
    def geometry(self) -> cuqi.geometry.Geometry:
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
        if self._target is not None:
            self.validate_target()

    # ------------ Public methods ------------
    def get_samples(self) -> Samples:
        """ Return the samples. The internal data-structure for the samples is a dynamic list so this creates a copy. """
        return Samples(np.array(self._samples).T, self.target.geometry)
    
    def reinitialize(self):
        """ Re-initialize the sampler. This clears the state and history and initializes the sampler again by setting state and history to their original values. """

        # Loop over state and reset to None
        for key in self._STATE_KEYS:
            setattr(self, key, None)

        # Loop over history and reset to None
        for key in self._HISTORY_KEYS:
            setattr(self, key, None)

        self._is_initialized = False

        self.initialize()
    
    def save_checkpoint(self, path):
        """ Save the state of the sampler to a file. """

        self._ensure_initialized()

        state = self.get_state()

        # Convert all CUQIarrays to numpy arrays since CUQIarrays do not get pickled correctly
        for key, value in state['state'].items():
            if isinstance(value, cuqi.array.CUQIarray):
                state['state'][key] = value.to_numpy()

        with open(path, 'wb') as handle:
            pkl.dump(state, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path):
        """ Load the state of the sampler from a file. """

        self._ensure_initialized()

        with open(path, 'rb') as handle:
            state = pkl.load(handle)

        self.set_state(state)

    def sample(self, Ns, batch_size=0, sample_path='./CUQI_samples/') -> 'Sampler':
        """ Sample Ns samples from the target density.

        Parameters
        ----------
        Ns : int
            The number of samples to draw.

        batch_size : int, optional
            The batch size for saving samples to disk. If 0, no batching is used. If positive, samples are saved to disk in batches of the specified size.

        sample_path : str, optional
            The path to save the samples. If not specified, the samples are saved to the current working directory under a folder called 'CUQI_samples'.

        """
        self._ensure_initialized()

        # Initialize batch handler
        if batch_size > 0:
            batch_handler = _BatchHandler(batch_size, sample_path)

        # Draw samples
        pbar = tqdm(range(Ns), "Sample: ")
        for idx in pbar:
            
            # Perform one step of the sampler
            acc = self.step()

            # Store samples
            self._acc.append(acc)
            self._samples.append(self.current_point)

            # display acc rate at progress bar
            pbar.set_postfix_str(f"acc rate: {np.mean(self._acc[-1-idx:]):.2%}")

            # Add sample to batch
            if batch_size > 0:
                batch_handler.add_sample(self.current_point)

            # Call callback function if specified            
            self._call_callback(idx, Ns)
                
        return self
    

    def warmup(self, Nb, tune_freq=0.1) -> 'Sampler':
        """ Warmup the sampler by drawing Nb samples.

        Parameters
        ----------
        Nb : int
            The number of samples to draw during warmup.

        tune_freq : float, optional
            The frequency of tuning. Tuning is performed every tune_freq*Nb samples.

        """

        self._ensure_initialized()

        tune_interval = max(int(tune_freq * Nb), 1)

        # Draw warmup samples with tuning
        pbar = tqdm(range(Nb), "Warmup: ")
        for idx in pbar:

            # Perform one step of the sampler
            acc = self.step()

            # Tune the sampler at tuning intervals
            if (idx + 1) % tune_interval == 0:
                self.tune(tune_interval, idx // tune_interval) 

            # Store samples
            self._acc.append(acc)
            self._samples.append(self.current_point)

            # display acc rate at progress bar
            pbar.set_postfix_str(f"acc rate: {np.mean(self._acc[-1-idx:]):.2%}")

            # Call callback function if specified
            self._call_callback(idx, Nb)

        return self
    
    def get_state(self) -> dict:
        """ Return the state of the sampler. 

        The state is used when checkpointing the sampler.

        The state of the sampler is a dictionary with keys 'metadata' and 'state'.
        The 'metadata' key contains information about the sampler type.
        The 'state' key contains the state of the sampler.

        For example, the state of a "MH" sampler could be:

        state = {
            'metadata': {
                'sampler_type': 'MH'
            },
            'state': {
                'current_point': np.array([...]),
                'current_target_logd': -123.45,
                'scale': 1.0,
                ...
            }
        }
        """   
        state = {
            'metadata': {
                'sampler_type': self.__class__.__name__
            },
            'state': {
                key: getattr(self, key) for key in self._STATE_KEYS
            }
        }
        return state

    def set_state(self, state: dict):
        """ Set the state of the sampler.

        The state is used when loading the sampler from a checkpoint.

        The state of the sampler is a dictionary with keys 'metadata' and 'state'.

        For example, the state of a "MH" sampler could be:

        state = {
            'metadata': {
                'sampler_type': 'MH'
            },
            'state': {
                'current_point': np.array([...]),
                'current_target_logd': -123.45,
                'scale': 1.0,
                ...
            }
        }
        """
        if state['metadata']['sampler_type'] != self.__class__.__name__:
            raise ValueError(f"Sampler type in state dictionary ({state['metadata']['sampler_type']}) does not match the type of the sampler ({self.__class__.__name__}).")
        
        for key, value in state['state'].items():
            if key in self._STATE_KEYS:
                setattr(self, key, value)
            else:
                raise ValueError(f"Key {key} not recognized in state dictionary of sampler {self.__class__.__name__}.")
            
    def get_history(self) -> dict:
        """ Return the history of the sampler. """
        history = {
            'metadata': {
                'sampler_type': self.__class__.__name__
            },
            'history': {
                key: getattr(self, key) for key in self._HISTORY_KEYS
            }
        }
        return history
    
    def set_history(self, history: dict):
        """ Set the history of the sampler. """
        if history['metadata']['sampler_type'] != self.__class__.__name__:
            raise ValueError(f"Sampler type in history dictionary ({history['metadata']['sampler_type']}) does not match the type of the sampler ({self.__class__.__name__}).")
        
        for key, value in history['history'].items():
            if key in self._HISTORY_KEYS:
                setattr(self, key, value)
            else:
                raise ValueError(f"Key {key} not recognized in history dictionary of sampler {self.__class__.__name__}.")

    # ------------ Private methods ------------
    def _call_callback(self, sample_index, num_of_samples):
        """ Calls the callback function. Assumes input is sampler, sample index, and total number of samples """
        if self.callback is not None:
            self.callback(self, sample_index, num_of_samples)

    def _validate_initialization(self):
        """ Validate the initialization of the sampler by checking all state and history keys are set. """

        for key in self._STATE_KEYS:
            if getattr(self, key) is None:
                raise ValueError(f"Sampler state key {key} is not set after initialization.")

        for key in self._HISTORY_KEYS:
            if getattr(self, key) is None:
                raise ValueError(f"Sampler history key {key} is not set after initialization.")
            
    def _ensure_initialized(self):
        """ Ensure the sampler is initialized. If not initialize it. """
        if not self._is_initialized:
            self.initialize()
            
    def _get_default_initial_point(self, dim):
        """ Return the default initial point for the sampler. Defaults to an array of ones. """
        return np.ones(dim)
    
    def __repr__(self):
        """ Return a string representation of the sampler. """
        if self.target is None:
            return f"Sampler: {self.__class__.__name__} \n Target: None"
        else:
            msg = f"Sampler: {self.__class__.__name__} \n Target: \n \t {self.target} "
            
        if self._is_initialized:
            state = self.get_state()
            msg += f"\n Current state: \n"
            # Sort keys alphabetically
            keys = sorted(state['state'].keys())
            # Put _ in the end
            keys = [key for key in keys if key[0] != '_'] + [key for key in keys if key[0] == '_']
            for key in keys:
                value = state['state'][key]
                msg += f"\t {key}: {value} \n"
        return msg

class ProposalBasedSampler(Sampler, ABC):
    """ Abstract base class for samplers that use a proposal distribution. """

    _STATE_KEYS = Sampler._STATE_KEYS.union({'current_target_logd', 'scale'})

    def __init__(self, target=None, proposal=None, scale=1, **kwargs):
        """ Initializer for abstract base class for samplers that use a proposal distribution.

        Any subclassing samplers should simply store input parameters as part of the __init__ method.

        Initialization of the sampler should be done in the _initialize method.

        See :class:`Sampler` for additional details.

        Parameters
        ----------
        target : cuqi.density.Density
            The target density.

        proposal : cuqi.distribution.Distribution, optional
            The proposal distribution. If not specified, the default proposal is used.

        scale : float, optional
            The scale parameter for the proposal distribution.

        **kwargs : dict
            Additional keyword arguments passed to the :class:`Sampler` initializer.

        """

        super().__init__(target, **kwargs)
        self.proposal = proposal
        self.initial_scale = scale

    def initialize(self):
        """ Initialize the sampler by setting and allocating the state and history before sampling starts. """

        if self._is_initialized:
            raise ValueError("Sampler is already initialized.")
        
        if self.target is None:
            raise ValueError("Cannot initialize sampler without a target density.")
        
        # Default values
        if self.initial_point is None:
            self.initial_point = self._get_default_initial_point(self.dim)

        if self.proposal is None:
            self.proposal = self._default_proposal

        # State variables
        self.current_point = self.initial_point
        self.scale = self.initial_scale

        self.current_target_logd = self.target.logd(self.current_point)

        # History variables
        self._samples = []
        self._acc = [ 1 ] # TODO. Check if we need to put 1 here.

        self._initialize() # Subclass specific initialization

        self._validate_initialization()

        self._is_initialized = True

    @abstractmethod
    def validate_proposal(self):
        """ Validate the proposal distribution. """
        pass     

    @property
    def _default_proposal(self):
        """ Return the default proposal distribution. Defaults to a Gaussian distribution with zero mean and unit variance. """
        return cuqi.distribution.Gaussian(np.zeros(self.dim), 1)

    @property
    def proposal(self):
        """ The proposal distribution. """
        return self._proposal
    
    @proposal.setter
    def proposal(self, proposal):
        """ Set the proposal distribution. """
        self._proposal = proposal
        if self._proposal is not None:
            self.validate_proposal()


class _BatchHandler:
    """ Utility class to handle batching of samples. 
    
    If a batch size is specified, this class will save samples to disk in batches of the specified size.
     
    This is useful for very large sample sets that do not fit in memory.
     
    """
    
    def __init__(self, batch_size=0, sample_path='./CUQI_samples/'):

        if batch_size < 0:
            raise ValueError("Batch size should be a non-negative integer")

        self.sample_path = sample_path
        self._batch_size = batch_size
        self.current_batch = []
        self.num_batches_dumped = 0

    @property
    def sample_path(self):
        """ The path to save the samples. """
        return self._sample_path
    
    @sample_path.setter
    def sample_path(self, value):
        if not isinstance(value, str):
            raise TypeError("Sample path must be a string.")
        normalized_path = value.rstrip('/') + '/'
        if not os.path.isdir(normalized_path):
            try:
                os.makedirs(normalized_path, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Could not create directory at {normalized_path}: {e}")
        self._sample_path = normalized_path

    def add_sample(self, sample):
        """ Add a sample to the batch if batching. If the batch is full, flush the batch to disk. """

        if self._batch_size <= 0:
            return  # Batching not used

        self.current_batch.append(sample)

        if len(self.current_batch) >= self._batch_size:
            self.flush()

    def flush(self):
        """ Flush the current batch of samples to disk. """

        if not self.current_batch:
            return  # No samples to flush

        # Save the current batch of samples
        batch_samples = np.array(self.current_batch)
        file_path = f'{self.sample_path}batch_{self.num_batches_dumped:04d}.npz'
        np.savez(file_path, samples=batch_samples, batch_id=self.num_batches_dumped)

        self.num_batches_dumped += 1
        self.current_batch = []  # Clear the batch after saving

    def finalize(self):
        """ Finalize the batch handler. Flush any remaining samples to disk. """
        self.flush()