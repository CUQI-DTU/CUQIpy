""" This module controls global modifiable configuration settings. """


MAX_DIM_INV = 2000
""" Max total dimension for expensive matrix computations (inversion, eigenvalues etc.) """

DEFAULT_SEED = 0
""" Default seed for random number generators. """

MAX_STACK_SEARCH_DEPTH = 1000
""" Maximum depth to search the Python variable stack for the name of a density if not set."""

MIN_DIM_SPARSE = 75
""" Minimum dimension to start storing Nd-arrays as sparse for N>2. The minimum dimension is defined as MIN_DIM_SPARSE^N. """

DISABLE_PROGRESS_BAR = False
""" Whether to disable progress bar during sampling. This is useful, for example, when building the pdf version of the gitbook (CUQI-Book) when bar printing spans many lines/pages."""
