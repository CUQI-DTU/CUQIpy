# %%Imports
import numpy as np
import cuqi 
from cuqi.model import Model
from cuqi.geometry import Continuous1D, Discrete, ConcatenatedGeometries,\
MappedGeometry
from cuqi.distribution import GMRF, Uniform, JointDistribution, Gaussian,\
    Normal, IndependentJointDistribution
from numbers import Number
import matplotlib.pyplot as plt

# %% Simple Example Supported
run_supported = False
# with one output only?
if run_supported:
    def fwd1(a):
        assert len(a) == 10
        return np.array([np.sum(a),np.prod(a)])
    geom_a = Continuous1D(np.linspace(0, 1, 10))
    range_geometry = Discrete(['y', 'z'])
    F1 = Model(forward=fwd1,
                range_geometry=range_geometry,
                domain_geometry=geom_a)
    a1 = np.random.randn(10)
    exact_data = F1(a1)
    assert np.allclose(exact_data, fwd1(a1))
    a = GMRF(0, 1, geometry=geom_a)
    y = Gaussian(mean=F1(a), cov = 0.001, geometry=range_geometry)
    y_given_abc = y(a=a1) 
    noisy_data = y_given_abc.sample(1)
    plt.figure()
    noisy_data.plot()
    noisy_data.geometry.plot(exact_data)
    plt.title("")
    joint = JointDistribution(a, y)
    posterior = joint(y=noisy_data)
    sampler = cuqi.sampler.MH(posterior)
    samples = sampler.sample_adapt(30)#30000
    plt.figure()
    samples.plot_ci(99.5, exact=a1)
    print(posterior.likelihood.geometry)

# %% Simple Example
def fwd3(a, b, c):
    assert len(a) == 10
    assert len(b) == 2
    #assert isinstance(c, Number)
    return b[0]*a + b[1]*c
geom_a = Continuous1D(np.linspace(0, 1, 10))
geom_b = Discrete(['b1', 'b2'])
geom_c = Discrete(['c'])
range_geometry = Continuous1D(np.linspace(0, 1, 10))
#*****Future*****:
#F = Model(forward=fwd3,
#          range_geometry=range_geometry,
#          domain_geometry=[geom_a, geom_b, geom_c])
F = Model(forward=fwd3,
            range_geometry=range_geometry,
            domain_geometry=
            ConcatenatedGeometries(geom_a, geom_b, geom_c))
#%%
a1 = np.random.randn(10)
b1 = np.random.randn(2)
c1 = 1
exact_data = F(a1, b1, c1)
exact_data2 = F(np.concatenate([a1, b1, [c1]]))
# What if any or all of the inputs are CUQIarray objects? will y_data be a CUQIarray object?
assert np.allclose(exact_data, fwd3(a1, b1, c1))
assert np.allclose(exact_data, exact_data2)
#%%
a = GMRF(0, 1, geometry=geom_a)
b = Normal(10, 0.1, geometry=geom_b)
c = Uniform(0, 1, geometry=geom_c)
y = Gaussian(mean=F(a, b, c), cov = 0.001, geometry=range_geometry)

#%%
y_given_abc = y(a=a1, b=b1, c=c1) 
noisy_data = y_given_abc.sample(1)
noisy_data.plot()
noisy_data.geometry.plot(exact_data)
#%%
joint = JointDistribution(a, b, c, y)
posterior = joint(y=noisy_data)
#%%
#*****Future*****:
# No need to explicitly create prior2 or use Posterior class
prior2 = IndependentJointDistribution(*posterior._distributions)
likelihood2 = posterior._likelihoods[0]
prior2.get_parameter_names()
#%%
posterior2 = cuqi.distribution.Posterior(likelihood2, prior2)
print(posterior2)
print(posterior2.dim)
posterior2.data.plot()
print(posterior2.geometry)
print(posterior2.logpdf(a1, b1, c1))
assert(posterior2.logpdf(a1, b1, c1)[0]==posterior.logd(a1, b1, c1)[0])
print(posterior2.get_conditioning_variables())
assert(posterior2.get_conditioning_variables()==[])
print(posterior2.get_parameter_names())
print(posterior2.model)



#%%
# raise error
try:
    posterior3 = cuqi.distribution.Posterior(likelihood2, a)
except ValueError as e:
    print("Error raised, error msg:")
    print(e.args[0])
#%%
sampler = cuqi.sampler.MH(posterior2)
#%%
samples = sampler.sample_adapt(100)
#%%
plt.figure()
samples['a'].plot_ci(99, exact=a1)
plt.figure()
samples['b'].plot_ci(99, exact=b1)
plt.figure()
samples['c'].plot_ci(99, exact=c1)
#%%
posterior2.likelihood.geometry
#%%
sampler_pCN = cuqi.sampler.pCN(posterior2)
#%%
samples_pCN = sampler_pCN.sample_adapt(10000)
#%%
plt.figure()
samples_pCN['a'].plot_ci(99, exact=a1)
plt.figure()
samples_pCN['b'].plot_ci(99, exact=b1)
plt.figure()
samples_pCN['c'].plot_ci(99, exact=c1)



############################################


# %% GEOMETRIES
range_geometry = Continuous1D(np.linspace(0, 1, 10))
geom1 = MappedGeometry(Continuous1D(np.linspace(0, 1, 10)),
                       lambda x: x+1,
                       lambda x: x-1)
geom2 = Discrete(['b'])
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
print("\nPrint: a_arr")
print(repr(a_arr))
print("\nPrint: a_arr.funvals")
print(repr(a_arr.funvals))
# 1. Another name could be CUQIarrayList or CUQIarrayTuple or ConcatenatedCUQIArray
# 2. The object can be indexed like a[0] or a[1]
# 3. similarly here, a.funvals returns
#    cuqi.array.CUQIarray(a[0].funvals, a[1].funvals, geometry=domain_geometry,
#        is_par=False)

a_arr2 = cuqi.array.CUQIarray(funval, geometry=domain_geometry, is_par=False)
print("\nPrint: a_arr2")
print(repr(a_arr2))
print("\nPrint: a_arr2.parameters")
print(repr(a_arr2.parameters))

# %%PRIORS
a = cuqi.distribution.GMRF(10, 0.1, geometry=cuqi.geometry.Continuous1D(np.linspace(0, 1, 10)))
b = cuqi.distribution.Uniform(0, 1)

prior = cuqi.distribution.JointDistribution(a, b)
prior_reduced = prior()
# or possibly another name

logpdf_val = prior.logd(vec1, 5) # returns a.logpdf(vec1) + b.logpdf(5)
print("\nPrint: logpdf of a_arr")
print(logpdf_val)

logpdf_val2 = prior_reduced.logd(vec1, 5) # returns a.logpdf(vec1) + b.logpdf(5)
print("\nPrint: logpdf of a_arr (reduced)")
print(logpdf_val2)
# %% Sampling
samples = prior_reduced.sample(10)
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

sample = prior_reduced.sample(1)
# generates 1 sample from each distribution and returns a ConcatenatedCUQIArray
# object




#%% test



# %%
from cuqi.distribution import *
gamma = Gamma(1,1)
gaussian1 = Gaussian(1, 1)
gaussian2 = Gaussian(1, lambda gamma:gamma)
joint = JointDistribution(gamma, gaussian2)
print(joint._get_conditioning_variables())
print(joint._internal_dependencies)
joint2 = JointDistribution(gamma, gaussian1)
try:
    joint1 = IndependentJointDistribution(gamma, gaussian2)
except ValueError as e:
    print("Error raised, error msg:")
    print(e.args[0])
#gaussian2.get_conditioning_variables()
# sample:
print(joint.is_independent)
print(joint2.is_independent)

# Reduce joint 2
joint2_reduced = joint2._reduce_to_single_density()
print(joint2_reduced)
joint2_reduced2 = joint2()
joint2_as_independent = IndependentJointDistribution(gamma, gaussian1)

#%% sample:
sample = joint2_reduced.sample(1)
sample.geometry

#%% logpdf
joint2_reduced.logpdf([1,1])
joint2_reduced.logpdf(1,1)

# %% sample with sampler
#%%
sampler = cuqi.sampler.MH(joint2_reduced)
samples = sampler.sample_adapt(10)


# %%


#%% Forward model
# %%Create a fwd function
def fwd(a,b):
    assert len(a) == 10
    return a + 2*b

F = cuqi.model.Model(fwd, range_geometry, domain_geometry)

#%%
# OR
try:
    F = cuqi.model.Model(fwd, 10, [1, 10]) #(2,2)
except TypeError as e:
    print("Error raised, error msg:")
    print(e.args[0])

#%%
vec3 = np.random.randn(11)
y_exact = F(vec3)
#%%
vec4 = np.random.randn(10)
y_exact = F(vec4, 1)
#%%
try:
    y_exact = F(1, np.random.randn(10))
except TypeError as e:
    print("Error raised, error msg:")
    print(e.args[0])
# In the second case, the model uses information from the geometry to
# to determine the first and the second argument
#%%
test_grad = False 
if test_grad:
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
y_given_abc = y(a=vec1, b=5)
y_data = y_given_abc.sample(1)

#%% Joint Distribution

#also
joint = cuqi.distribution.JointDistribution(a, b, y)

#%% Posterior
posterior = joint(y=y_data)
#%% Sampling the posterior
try:
    # Need to be fixed
    sampler = cuqi.sampler.MH(posterior)
except TypeError as e:
    print("Error raised, error msg:")
    print(e.args[0])
# sampler uses the par_dim to create a proposal of size (11,)
# model can accept input of size  (11,)

#%% Update gradient chain rule in model to consider chain from two distributions