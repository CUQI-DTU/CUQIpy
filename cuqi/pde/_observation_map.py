import scipy
import numpy as np
"""
This module contains observation map examples for PDE problems. The map can
be passed to the `PDE` object initializer via the `observation_map` argument.

For example on how to use set observation maps in time dependent PDEs, see
`demos/howtos/time_dependent_linear_pde.py`.
"""

# 1. Steady State Observation Maps
# --------------------------------

# 2. Time-Dependent Observation Maps
# -----------------------------------
def FD_spatial_gradient(sol, grid, times):
    """Time dependent observation map that computes the finite difference (FD) spatial gradient of a solution given at grid points (grid) and times (times). This map is supported for 1D spatial domains only.
    
    Parameters
    ----------
    sol : np.ndarray
        The solution array of shape (number of grid points, number of time steps).

    grid : np.ndarray
        The spatial grid points of shape (number of grid points,).

    times : np.ndarray
        The discretized time steps of shape (number of time steps,)."""

    if len(grid.shape) != 1:
        raise ValueError("FD_spatial_gradient only supports 1D spatial domains.")
    observed_quantity = np.zeros((len(grid)-1, len(times)))
    for i in range(observed_quantity.shape[0]):
        observed_quantity[i, :] = ((sol[i, :] - sol[i+1, :])/
                                       (grid[i] - grid[i+1]))
    return observed_quantity