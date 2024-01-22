# %%Imports
import numpy as np
import cuqi 
from cuqi.geometry import Continuous1D, Discrete, ConcatenatedGeometries,\
MappedGeometry

# %% GEOMETRIES
range_geometry = Continuous1D(np.linspace(0, 1, 10))
geom1 = MappedGeometry(Continuous1D(np.linspace(0, 1, 10)),
                       lambda x: x+1,
                       lambda x: x-1)
geom2 = Discrete(['a'])
domain_geometry = ConcatenatedGeometries( geom1, geom2)
# Note: 
#1. direct product implements functions such as par2fun, fun2par, plot, 
#   gradient, etc, by calling each function on each component of the direct
#   product
#2. another name choice could be ConcatenatedGeometries

print(domain_geometry.par_shape) # returns [(10,), (1,)]
print(domain_geometry.fun_shape) # returns [(10,), (1,)]
print(domain_geometry.par_dim) # returns 11
print(domain_geometry.fun_dim) # returns [(10,), 1]
                        # how to distinguish between 2D geometry and direct product?

vec1 = np.random.randn(10)
vec2 = np.random.randn(11)

funval = domain_geometry.par2fun(vec2)

par = domain_geometry.fun2par(funval[0], funval[1])
# par is a numpy array of shape (11,)
# this format is helpful for samplers


#%% Assert values are equal
assert np.allclose(funval[0], geom1.par2fun(vec2[:geom1.par_dim]))
assert np.allclose(funval[1], geom2.par2fun(vec2[geom1.par_dim:]))
assert np.allclose(par[:geom1.par_dim], geom1.fun2par(funval[0]))
assert np.allclose(par[geom1.par_dim:], geom2.fun2par(funval[1]))

# %% CUQIarray
a_arr = cuqi.array.CUQIarray(vec2, geometry=domain_geometry)
print("a_arr")
print(repr(a_arr))
print("a_arr.funvals")
print(repr(a_arr.funvals))
# 1. Another name could be CUQIarrayList or CUQIarrayTuple or ConcatenatedCUQIArray
# 2. The object can be indexed like a[0] or a[1]
# 3. similarly here, a.funvals returns
#    cuqi.array.CUQIarray(a[0].funvals, a[1].funvals, geometry=domain_geometry,
#        is_par=False)

a_arr2 = cuqi.array.CUQIarray(funval, geometry=domain_geometry, is_par=False)
print("a_arr2")
print(repr(a_arr2))
print("a_arr2.parameters")
print(repr(a_arr2.parameters))

# %%PRIORS
a = cuqi.distribution.Uniform(0, 1)
b = cuqi.distribution.GMRF(10, 0.1, geometry=cuqi.geometry.Continuous1D(np.linspace(0, 1, 10)))

prior = cuqi.distribution.IndependentJointDistribution(a, b)
# or possibly another name
prior = cuqi.distribution.JointPrior(a, b)

logpdf_val = prior.logpdf(vec1, 5) # returns a.logpdf(vec1) + b.logpdf(5)

samples = prior.sample(10)
# 1. generates 10 samples from each distribution in the 
# returned object
# 2. The returned object could be a ConcatenatedSamples object
#    - can be indexed like samples[0] or samples[1]
#    - samples.some_function will call some_function on each component
#    - samples.some_plot_function() will plot each component separately
#    - samples.funvals will return 
#          ConcatenatedSamples(samples[0].funvals,
#                              samples[1].funvals,
#                              geometry=domain_geometry,
#                              is_par=False)
# 3. Can think of better name for IndependentJointDistribution

sample = prior.sample(1)
# generates 1 sample from each distribution and returns a ConcatenatedCUQIArray
# object

#%% Forward model
# %%Create a fwd function
def fwd(a,b):
    assert len(b) == 10
    return a + 2*b

F = cuqi.model.Model(fwd, range_geometry, domain_geometry)
# OR
F = cuqi.model.Model(fwd, 10, [1, 10]) #(2,2)

y_exact = F(1, np.random.randn(10))
y_exact = F(np.random.randn(11))

# In the second case, the model uses information from the geometry to
# to determine the first and the second argument

# The model can be created with a gradient 
def grad(a, b, direction):
    pass
F = cuqi.model.Model(fwd, gradient=grad)
# in this case the gradient returns a numpy array of shape (11) or 
# a list [array of shape (10,), array of shape (1,)]

#Another option (possibly support both)
def grad_a(a, b, direction):
    pass
def grad_b(a, b, direction):
    pass
F = cuqi.model.Model(fwd, gradient=[grad_a, grad_b])

#%% Data Distribution
y = cuqi.distribution.Gaussian(F(a, b), 0.1, geometry=range_geometry)
# Distributions to be extended to enable multiple inputs to the forward model
y_data = y(a=5, b=vec1).sample(1)

#%% Joint Distribution
joint = cuqi.distribution.JointDistribution(prior, y)
#also
joint = cuqi.distribution.JointDistribution(a, b, y)

#%% Posterior
posterior = joint.set_data(y_data)
#%% Sampling the posterior
sampler = cuqi.sampler.MH(posterior)
# sampler uses the par_dim to create a proposal of size (11,)
# model can accept input of size  (11,)

#%% Update gradient chain rule in model to consider chain from two distributions
