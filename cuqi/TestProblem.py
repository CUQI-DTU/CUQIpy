import numpy as np
from scipy.sparse import csc_matrix
from scipy.integrate import quad_vec


#=============================================================================
class Deblur(object):
    
    def __init__(self, a = 48, noise_std = 0.1, dim = 128, bnds = [0, 1]):
        self.dim = dim
        self.t = np.linspace(bnds[0], bnds[1], dim)
        self.h = self.t[1] - self.t[0]

        # set-up computational model kernel
        self.a = a
        kernel = lambda x, y, a: a / 2*np.exp(-a*abs((x-y)))   # blurring kernel

        # convolution matrix
        T1, T2 = np.meshgrid(self.t, self.t)
        A = self.h*kernel(T1, T2, a)
        maxval = A.max()
        A[A < 5e-3*maxval] = 0
        self.A = csc_matrix(A)   # make A sparse

        # forward model
        # self.forward = lambda x: A @ x

        # Generate inverse-crime free data
        self.sigma_obs = noise_std
        self.corrmat = np.eye(dim)
        self.data, self.f_true, self.g_true = self.data_conv(kernel)

    def forward(self, x):
        return self.A @ x

    def data_conv(self, kernel):
        np.random.seed(1)

        # f is piecewise constant
        x_min, x_max = self.t[0], self.t[-1]
        vals = np.array([0, 2, 3, 2, 0, 1, 0])
        conds = lambda x: [(x_min <= x) & (x < 0.1), (0.1 <= x) & (x < 0.15), (0.15 <= x) & (x < 0.2),  \
                   (0.20  <= x) & (x < 0.25), (0.25 <= x) & (x < 0.3), (0.3 <= x) & (x < 0.6), \
                   (0.6 <= x) & (x <= x_max)]
        f_signal = lambda x: np.piecewise(x, conds(x), vals)

        # numerically integrate the convolution
        a_true = 50
        g_conv = lambda x: quad_vec(lambda y: f_signal(y)*kernel(x, y, a_true), x_min, x_max)
        # se also np.convolve(kernel(...), f_true, mode='same')

        # true values
        f_true = f_signal(self.t)
        g_true = g_conv(self.t)[0]

        # noisy data
        b = g_true + np.random.normal(loc=0, scale=self.sigma_obs, size=(self.dim))

        return b, f_true, g_true
