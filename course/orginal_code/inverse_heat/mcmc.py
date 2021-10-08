import scipy.stats as scp
import numpy as np
import matplotlib
import progressbar as progress

class Random_Walk:
    def __init__(self,pi_like,x0):
        self.pi_like = pi_like
        self.x0 = x0
        self.ndim = self.x0.shape[0]

        self.target_last = pi_like(self.x0) - ( 0.5*np.sum(x0**2) )
        self.sample_last = self.x0

        self.all_samples = [ self.x0 ]
        self.all_targets = [ self.target_last ]
        self.all_accepts = [ 1 ]

        pi = lambda p: self.pi_like( p ) - ( 0.5*np.sum(p**2) )
        constraint = lambda p: True
        self.update = Metropolis_update( pi, self.sample_last, constraint )

    def sample(self, Ns, Nb):
        for s in progress.progressbar( range(Ns+Nb-1) ):
            self.single_step()

    def single_step(self):
        sample, target, acc = self.update.step(0.05)

        self.sample_last = sample
        self.target_last = target

        self.all_samples.append( sample )
        self.all_targets.append( target )
        self.all_accepts.append( acc )

    def print_stat(self):
        print( np.array( self.all_accepts ).mean() )

    def give_stats(self):
        return np.array(self.all_samples), np.array(self.all_targets)

class Metropolis_update:
    def __init__(self, pi_target,x_old,constraint):
        self.pi_target = pi_target
        self.x_old = x_old
        self.dim = self.x_old.shape[0]
        self.constraint = constraint
        self.target_old = self.pi_target( self.x_old )

    def set_pi_target(self, pi_target):
        self.pi_target

    def step(self, beta):
        flag = False
        while( flag == False ):
            x = self.x_old + beta*np.random.standard_normal(self.dim)
            flag = self.constraint(x)

        target = self.pi_target( x )
        ratio = np.exp(target - self.target_old)
        alpha = min(1., ratio)
        uu = np.random.uniform(0,1)
        if (uu <= alpha):
            acc = 1
            self.x_old = x
            self.target_old = target
        else:
            acc = 0

        return self.x_old, self.target_old, acc