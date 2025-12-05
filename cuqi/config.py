""" This module controls global modifiable configuration settings. """


MAX_DIM_INV = 2000
""" Max total dimension for expensive matrix computations (inversion, eigenvalues etc.) """

DEFAULT_SEED = 0
""" Default seed for random number generators. """

MAX_STACK_SEARCH_DEPTH = 1000
""" Maximum depth to search the Python variable stack for the name of a density if not set."""

MIN_DIM_SPARSE = 75
""" Minimum dimension to start storing Nd-arrays as sparse for N>2. The minimum dimension is defined as MIN_DIM_SPARSE^N. """

PROGRESS_BAR_STATS_DYNAMIC_UPDATE = True
""" If True, sampling progress bar statistics such as acceptance rate are
updated frequently (dynamic updates).
If False, progress bars are minimal, only showing default stats, e.g., iteration
number and time elapsed. Setting this configuration variable to False can be
useful, for example, when building documentation or the Jupyter book
(CUQI-Book). This is because the documentation or book build systems may not
handle dynamic updates of non-default progress bar stats well, leading to many
lines of output.
"""