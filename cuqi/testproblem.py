import numpy as np
from scipy.sparse import csc_matrix
from scipy.integrate import quad_vec

from cuqi.model import LinearModel
from cuqi.distribution import Gaussian
from cuqi.problem import Type1

#=============================================================================
class Deblur(Type1):
    
    def __init__(self, a = 48, noise_std = 0.1, dim = 128, bnds = [0, 1]):
        t = np.linspace(bnds[0], bnds[1], dim)
        h = t[1] - t[0]

        # set-up computational model kernel
        kernel = lambda x, y, a: a / 2*np.exp(-a*abs((x-y)))   # blurring kernel

        # convolution matrix
        T1, T2 = np.meshgrid(t, t)
        A = h*kernel(T1, T2, a)
        maxval = A.max()
        A[A < 5e-3*maxval] = 0
        A = csc_matrix(A)   # make A sparse

        # Store forward model
        model = LinearModel(lambda x: A @ x,lambda y: A.T @ y, dim=np.shape(A))
        
        # Store Noise model
        noise = Gaussian(np.zeros(dim),noise_std,np.eye(dim))
        
        # Generate inverse-crime free data
        data, f_true, g_true = data_conv(t,kernel,noise)
        
        #Initialize deblur as Type1 cuqi probler
        Type1.__init__(self,data,model,noise,[]) #No default prior
        
        #Store other properties
        self.meshsize = h
        self.f_true = f_true
        self.g_true = g_true
        self.t = t
        

def data_conv(t,kernel,noise):
        np.random.seed(1)

        # f is piecewise constant
        x_min, x_max = t[0], t[-1]
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
        f_true = f_signal(t)
        g_true = g_conv(t)[0]

        # noisy data
        b = g_true + np.squeeze(noise.sample(1)) #np.random.normal(loc=0, scale=self.sigma_obs, size=(self.dim))

        return b, f_true, g_true