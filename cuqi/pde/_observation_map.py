import scipy
import numpy as np

def spatial_gradient(sol, grid, times):
    """Observation map that computes the spatial gradient of the solution given
    at grid points (grid) and times (times)."""
    solution_obs = np.zeros((len(grid)-1, len(times)))
    for i in range(solution_obs.shape[0]):
        solution_obs[i, :] = ((sol[i, :] - sol[i+1, :])/
                                       (grid[i] - grid[i+1]))
    return solution_obs

def _extract_spatial_temporal_obs(sol, grid, times, grid_obs, time_obs):
    """Private function to extract solution at observation points in space and
    time."""
    # Interpolate solution in space and time to the observation
    # time and space
    solution_obs = scipy.interpolate.RectBivariateSpline(
        grid, times, sol)(grid_obs, time_obs)
    return solution_obs