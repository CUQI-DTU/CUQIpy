import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from mcmc import Random_Walk

# this class solves the heat equation with FD method
# d^2/dx^2 u = f,  x in [0,pi]
# u(0,t) = u(pi,t) = 0
# u(x,0) = u0
class heat:
    # class init func setting up FD method
    def __init__(self):
        self.N = 128 # number of discretization points
        self.dx = np.pi/(self.N+1) # space step size

        self.x = np.linspace(self.dx,np.pi,self.N,endpoint=False)
        self.init_cond = rand_field(self.N) # definig the initial condition to be a Gaussian random field
        self.u0 = self.init_cond.rand_func() 

        cfl= 5/11 # the cfl condition to have a stable solution
        self.dt = cfl*self.dx**2 # defining time step
        self.T = .2 # defining the maximum time
        self.MAX_ITER = int(self.T/self.dt) # number of time steps

        self.Dxx = np.diag( (1-2*cfl)*np.ones(self.N) ) + np.diag(cfl* np.ones(self.N-1),-1) + np.diag(cfl*np.ones(self.N-1),1) # FD diffusion operator

    def set_init_cond(self, u0):
        self.u0 = u0

    # solves the heat equation and shows the solution
    def time_stepping(self):
        self.sol = [ np.copy( self.u0 ) ]
        u_old = np.copy( self.u0 )
        for i in range(self.MAX_ITER):
            u_old = self.Dxx@u_old

            self.sol.append(u_old)

        self.sol = np.array(self.sol)

        T = np.array( range(0,self.MAX_ITER+1) )
        (X,T) = np.meshgrid(self.x,T)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(T,X,self.sol)
        plt.show()

    # computes the solution at t=0.2 for a given initial condition
    def advance_with_init_cond(self,u0):
        u_old = u0
        for i in range(self.MAX_ITER):
            u_old = self.Dxx@u_old

        return u_old   

    def advance_time(self):
        u_old = np.copy( self.u0 )
        for i in range(self.MAX_ITER):
            u_old = self.Dxx@u_old

        return u_old

    # computes the solution at t=0.2 for a given expansion coefficients
    def forward(self,p):
        self.init_cond.set_params(p) # set the expansion coefficients
        self.u0 = self.init_cond.give_real_func() # compute the real initial condition from expansion coefficients
        return self.advance_time()


# class representation of the random field in  the sine basis
# alpha = sum_i p * (1/i)^decay * sin(ix)
class rand_field:
    # init function defining paramters for the KL expansion
    def __init__(self,N):
        self.N = N # number of modes
        self.modes = np.zeros(N) # vector of expansion coefs
        self.real = np.zeros(N) # vector of real values
        self.decay_rate = 2.5 # decay rate of KL
        self.c = 12. # normalizer factor
        self.coefs = np.array( range(1,self.N+1) ) # KL eigvals
        self.coefs = 1/np.float_power( self.coefs,self.decay_rate )

        self.p = np.zeros(self.N) # random variables in KL

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
        return self.real

# this subroutine solves the inverse problem
#   y = G(p) + e
#   - G(p) is the forward map that maps expansion 
#   coefficients to the solution of the heat equation
#   - e is the Gaussian observation noise with covariance
#   sigma^2*I
#   - y is the noisy observation
def inverse_heat():
    # computing the true initial condition
    # alpha = x * exp(-2x) * sin(pi-x)
    N = 128
    dx = np.pi/(N+1)
    x = np.linspace(dx,np.pi,N,endpoint=False)
    true_init = x*np.exp(-2*x)*np.sin(np.pi-x)

    # defining the heat equation as the forward map
    prob = heat()
    y_obs = prob.advance_with_init_cond(true_init) # observation vector

    SNR = 100 # signal to noise ratio
    sigma = np.linalg.norm(y_obs)/SNR
    sigma2 = sigma*sigma # variance of the observation Gaussian noise

    # formulating the Bayesian inverse problem
    pi_like = lambda p: - ( 0.5*np.sum(  (prob.forward(p) - y_obs)**2)/sigma2) # least squares likelihood
    p0 = np.zeros(N) # initial parameter guess
    Ns = 10000 # number of samples
    Nb = 10 # number of burn-in samples
    RWM = Random_Walk(pi_like,p0) # type of MCMC (Metropolis)
    RWM.sample(Ns,Nb) # running MCMC

    RWM.print_stat() # this prints the acceptance rate
    samples, target = RWM.give_stats() # extracting samples

    # computing the mean parameter
    p_mean = np.mean(samples[300:,:],axis=0)

    # computing the mean field
    field = rand_field(N)
    field.set_params(p_mean)
    f = field.give_real_func()

    # plotting the mean solution vs the true solution
    fig, ax1 = plt.subplots(1)
    ax1.plot(x,f,label=r'posterior mean',color="blue",linewidth=2.)
    ax1.plot(x,true_init,label=r'true initial condition',color="orange",linewidth=2.)
    plt.show()

if __name__ == "__main__":
    inverse_heat()