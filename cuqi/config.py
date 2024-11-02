""" This module controls global modifiable configuration settings. """


MAX_DIM_INV = 2000
""" Max total dimension for expensive matrix computations (inversion, eigenvalues etc.) """

DEFAULT_SEED = 0
""" Default seed for random number generators. """

MAX_STACK_SEARCH_DEPTH = 1000
""" Maximum depth to search the Python variable stack for the name of a density if not set."""

MIN_DIM_SPARSE = 75
""" Minimum dimension to start storing Nd-arrays as sparse for N>2. The minimum dimension is defined as MIN_DIM_SPARSE^N. """

USE_MULTIPROCESSING = False
""" Use multiprocessing for parallel computations. This is mainly used for parallelizing the computation of the gradient of the log density function when finite difference approximation is enabled. """

NUM_PROCESSORS = 5
""" Number of processors to use for parallel computations. """
