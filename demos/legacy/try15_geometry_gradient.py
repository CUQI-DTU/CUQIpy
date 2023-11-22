#%% Import required libraries
import sys
sys.path.append("..")
from cuqi.geometry import Continuous1D
import numpy as np
import cuqi

#%% Set up a cuqi model with simple geometries
model = cuqi.model.Model(forward=lambda x: np.sin(x),
                         range_geometry=Continuous1D(3),
                         domain_geometry=Continuous1D(3),
                         gradient=lambda direction, x: np.diag(
                             np.cos(x))@direction,
                         )

#%% evaluate CUQI model gradient at point `wrt` and direction `direction`
wrt = np.array([1, 1, 4])
direction = np.array([1, 12, 8])

grad = model.gradient(direction, wrt)
print("CUQI model gradient:", grad)

#%% Compute a finite difference approximation of the cuqi model gradient
findiff_grad = cuqi.utilities.approx_derivative(
    model.forward, wrt, direction)
print("Approximate gradient:", findiff_grad)

#%% Change the model geometry to a mapped domain geometry
model.domain_geometry = cuqi.geometry.MappedGeometry(Continuous1D(
    3), map=lambda x: x**2, imap=lambda x: np.sqrt(x))

#%% Set the mapped domain geometry gradient
model.domain_geometry.gradient = lambda direction, x: 2*np.diag(x)@direction

#%% Check the mapped function value for the input `wrt`
print("`wrt`: ",wrt )
print("mapped `wrt`: ",model.domain_geometry.par2fun(wrt) )


#%% evaluate CUQI model gradient at point `wrt` and direction `direction`
grad = model.gradient(direction, wrt)
print("CUQI model gradient:", grad)

#%% Compute a finite difference approximation of the cuqi model gradient
findiff_grad = cuqi.utilities.approx_derivative(
    model.forward, wrt, direction)
print("Approximate gradient:", findiff_grad)
