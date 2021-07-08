#%% Import required libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
import cuqi

try: 
    import dolfin as dl
    import ufl
except Exception as error:
    warnings.warn(error.msg)

import numpy as np
from dolfin import *

# This class models a log-Gaussian random field
# The parent class UserExpression is used in dolfin-FEniCS
# to define user specific functions in the FEM formulation
class Matern(dl.UserExpression):
    # defining the type of the random field
    def set_type(field_type="log_gaussian"):
        self.field_type = field_type

    # setting the KL parameters
    def set_field_params(self,gamma=2.,KL_terms=10,mean_value=1.):
        self.gamma = gamma
        self.KL_terms = KL_terms
        self.mean_value = mean_value

    # setting Gaussian parameters in the KL expansion
    def set_prior_params(self,p):
        self.a = p[:self.KL_terms]
        self.b = p[self.KL_terms:]

    # eval function is required for FEniCS
    def eval(self, values, x):
        total_sum = self.mean_value
        for i in range(0,self.KL_terms):
            total_sum += ( self.a[i]*np.sin( 2*np.pi*x[0] ) + self.b[i]*np.cos( 2*np.pi*x[0] ) ) / np.power( i+1, self.gamma )
        values[0] = np.exp( total_sum )


#%% Set up and solve cuqi problem that uses FEniCSPDEModel 
# Problem of form b = A(x) + e
# Create operator A

mesh = dl.UnitIntervalMesh(100) 

Vh_STATE = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh_PARAMETER = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=20)
Vh_ADJOINT = dl.FunctionSpace(mesh, 'Lagrange', 1)
Vh = [Vh_STATE, Vh_PARAMETER, Vh_ADJOINT]

def u_boundary(x, on_boundary):
    return on_boundary and x[0]<dl.DOLFIN_EPS

u_bdr = dl.Expression("0", degree=1)
bc = dl.DirichletBC(Vh_STATE, u_bdr, u_boundary)

u_bdr0 = dl.Constant(0.0)
bc0 = dl.DirichletBC(Vh_ADJOINT, u_bdr0, u_boundary)

f_exp = dl.Expression( "10*exp( -(pow(x[0] - 0.5, 2) ) / 0.02)" , degree = 1)
nbc_exp = dl.Expression("(x[0]>.7)*sin( 5*x[0] )", degree = 1)

def form(u,m,p):
    alpha = Matern(element = u.ufl_element())
    alpha.set_field_params(gamma=2.,KL_terms=10,mean_value=1.)
    alpha.set_prior_params(m.vector().get_local())
    return alpha*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f_exp*p*ufl.dx - nbc_exp*p*ufl.ds

def obs_op(m, u):
    return u(1)


A = cuqi.PDEmodel.FEniCSPDEModel(form, mesh, Vh, bc=bc, bc0=bc0, f=None, obs_op=obs_op)

#%% Create & plot prior 
prior = cuqi.distribution.Gaussian(np.zeros(20), np.ones(20))
ps = prior.sample(100)
plt.figure()
ps.plot_ci(95,exact=np.zeros(20))

#%% Create noise (e) and data (b)
#obs_data = np.load('obs.npz')
#    y_obs = obs_data['obs']


#true_m = prior.sample(1)
#true_b = A.forward(true_m)
#noise_std = 0.01 * np.max(true_b)
#n = 1
#e = cuqi.distribution.Gaussian(np.array([0]), noise_std, np.eye(n))
#b = true_b + e.sample()[0]

n =1 
b = np.array([np.load('../../cuqisandbox/Babak/obs.npz')['obs'] ])
sigma = np.abs( b )/100
e = cuqi.distribution.Gaussian(np.array([0]), sigma, np.eye(n))

#%% Create cuqi problem (sets up likelihood and posterior) 
IP = cuqi.problem.Type1(b,A,e,prior)

#%% Sample & plot posterior
true_m = np.array([np.load('../../cuqisandbox/Babak/obs.npz')['true_params'] ]).T
results = IP.sample(Ns=10000) 
plt.figure()
results.plot_ci(95,exact=true_m)
plt.show()


alpha = Matern(element = Vh_STATE.ufl_element())
alpha.set_field_params(gamma=2.,KL_terms=10,mean_value=1.)
alpha.set_prior_params(true_m)
a = dl.Function(Vh_STATE)
a.interpolate(alpha)
dl.plot(a)

alpha.set_prior_params(np.mean(results.samples,axis=1))
a.interpolate(alpha)
dl.plot(a)