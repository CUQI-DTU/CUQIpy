import numpy as np
from scipy.fftpack import dst, idst
from scipy.linalg import solve
from mcmc import Random_Walk
import matplotlib.pyplot as plt
from geometry import KLField

# class representation of the random field in  the sine basis
# alpha = sum_i p * (1/i)^decay * sin(ix)
class gauss_field:
    # init function defining paramters for the KL expansion
    def __init__(self,N):
        self.N = N+1 # number of modes
        self.modes = np.zeros(N) # vector of expansion coefs
        self.real = np.zeros(N) # vector of real values
        self.decay_rate = 2.5 # decay rate of KL
        self.c = 12. # normalizer factor
        self.amp_factor = 10. # amplification factor
        self.coefs = np.array( range(1,self.N+1) ) # KL eigvals
        self.coefs = 1/np.float_power( self.coefs,self.decay_rate )

        self.p = np.zeros(self.N) # random variables in KL
        #self.x = np.linspace(0,1,N+2)

    # sets values for KL random variables
    def set_params(self,p):
        self.p = p

    # random realization of the field
    def rand_func(self):
        self.p = np.random.standard_normal( self.N )
        self.modes = self.p*self.coefs/self.c
        self.real = idst(self.modes)/2
        return self.real

    # computes the real function out of expansion coefs
    def give_real_func(self):
        self.modes = self.p*self.coefs/self.c
        self.real = idst(self.modes)/2
        return np.exp( self.amp_factor*self.real )

# the class dealing with FD for the Poisson equation on a staggered grid
# d/dx ( alpha du/dx ) = f
# u = 0. at x=0 and x=1
class poisson:
    # init function computing FD matrices
    def __init__(self,N):
        self.N = N # number of FD nodes
        self.dx = 1./self.N # step size
        self.Dx =  - np.diag( np.ones(N) , 0 ) + np.diag( np.ones(N-1) , 1 )
        vec = np.zeros(N)
        vec[0] = 1
        self.Dx = np.concatenate( [ vec.reshape( [1,-1] ), self.Dx], axis = 0 )
        self.Dx /= self.dx # FD derivative matrix

        #self.alpha = gauss_field(self.N) # conductivity field
        self.alpha = KLField(self.N+1)

        x = np.linspace( self.dx,1.,N, endpoint=False )
        self.f = 10*np.exp( -( (x - 0.5)**2 ) / 0.02) # source term

    # function computing observed function
    def solve_with_conductivity(self, true_alpha):
        Dxx = self.Dx.T@np.diag(true_alpha)@self.Dx
        u = solve(Dxx,self.f)
        return u

    # the forward map from field parameters to the solution
    def forward(self,p):
        a = np.exp( 10*self.alpha.to_function(p) )
        Dxx = self.Dx.T@np.diag(a)@self.Dx
        return solve(Dxx,self.f)

# this subroutine solves the inverse problem
#   y = G(p) + e
#   - G(p) is the forward map that maps expansion 
#   coefficients to the solution of the solution of the
#   Poisson equation
#   - e is the Gaussian observation noise with covariance
#   sigma^2*I
#   - y is the noisy observation
def inverse_poisson():
    # computing the true conductivity
    # alpha = exp( 5 * x * exp(-2x) * sin(pi-x) )
    N = 128
    dx = np.pi/N
    x = np.linspace(dx/2,np.pi-dx/2,N+1)
    true_alpha = np.exp( 5*x*np.exp(-2*x)*np.sin(np.pi-x) )

    # defining the poisson equation as the forward map
    problem = poisson(N)
    y_obs = problem.solve_with_conductivity(true_alpha)

    SNR = 100 # signal to noise ratio
    sigma = np.linalg.norm(y_obs)/SNR
    sigma2 = sigma*sigma # variance of the observation Gaussian noise

    # formulating the Bayesian inverse problem
    pi_like = lambda p: - ( 0.5*np.sum(  (problem.forward(p) - y_obs)**2)/sigma2) # least squares likelihood
    p0 = np.zeros(N+1) # initial parameter guess
    Ns = 50000 # number of samples
    Nb = 10 # number of burn-in samples
    RWM = Random_Walk(pi_like,p0) # type of MCMC (Metropolis)
    RWM.sample(Ns,Nb) # running MCMC

    RWM.print_stat() # this prints the acceptance rate
    samples, target = RWM.give_stats() # extracting samples

    # computing the mean parameter
    p_mean = np.mean(samples[300:,:],axis=0)

    # computing the mean field
    field = gauss_field(N)
    field.set_params(p_mean)
    f = field.give_real_func()

    # plotting the mean solution vs the true solution
    fig, ax1 = plt.subplots(1)
    ax1.plot(x,f,label=r'posterior mean',color="blue",linewidth=2.)
    ax1.plot(x,true_alpha,label=r'true initial condition',color="orange",linewidth=2.)
    plt.show()

def script():
    N = 128
    problem = poisson(N)


if __name__ == "__main__":
    inverse_poisson()
    #script()