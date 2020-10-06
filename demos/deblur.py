# =============================================================================
# Created by:
# Felipe Uribe @ DTU
# =============================================================================
# Version 2020-02
# =============================================================================
import numpy as np
import scipy as sp
import scipy.integrate


#=============================================================================
def deblur(t_mid, kernel, sigma_obs, n):
    np.random.seed(1)

    # f is piecewise constant
    x_min, x_max = 0, 1
    vals = np.array([0, 2, 3, 2, 0, 1, 0])
    conds = lambda x: [(x_min <= x) & (x < 0.1), (0.1 <= x) & (x < 0.15), (0.15 <= x) & (x < 0.2),  \
                   (0.20  <= x) & (x < 0.25), (0.25 <= x) & (x < 0.3), (0.3 <= x) & (x < 0.6), \
                   (0.6 <= x) & (x <= x_max)]
    f_signal = lambda x: np.piecewise(x, conds(x), vals)
    
    # numerically integrate the convolution
    a_true = 50
    g_conv = lambda x: sp.integrate.quad_vec(lambda y: f_signal(y)*kernel(x,y,a_true), x_min, x_max)
    # se also np.convolve(kernel(...), f_true, mode='same')
    
    # true values
    f_true = f_signal(t_mid)
    g_true = g_conv(t_mid)[0]
    
    # noisy data
    b = g_true + np.random.normal(loc=0, scale=sigma_obs, size=(n))

    return b, f_true, g_true