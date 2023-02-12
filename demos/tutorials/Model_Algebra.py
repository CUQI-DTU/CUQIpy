"""
Model Algebra
=============

CUQIpy supports algebraic operations on models. This is done by defining
two or more models and then combining them using the algebraic operators
+ (addition), - (subtraction), * (multiplication) and / (division).

Currently, the following operations are supported:

- Addition of models

"""

# %%
# First we import the necessary modules
import cuqi

# %%
# The sum of two models
# ---------------------
# 
# Suppose we have two models A: x -> z and B: y -> z
# In this case we consider a regular Model and a LinearModel

A = cuqi.model.Model(lambda x: x**2, 1, 1)
B = cuqi.model.LinearModel(lambda y: 5*y, lambda z: 5*z, 1, 1)

# %%
# The sum linear model J: (x, y) -> z is then defined
# as the sum of the two models A and B, i.e. J(x, y) = A(x) + B(y)
# In CUQIpy this is done by passing the two models to the SumModel class

J = A + B


# %%
# We can evaluate the model simply by calling it with specific inputs
# In this case both A and B are identity functions, so J(x, y) = x^2 + 5*y
# In this case J(x=1, y=2) = 1^2 + 5*2 = 11

J(x=1, y=2)

# %%
# We can also partially evaluate the model
# This returns a new model that is a function of the remaining input
# in this case J(x=1) gives a new model taking y -> 1^2 + 5*y.
# Notice, in this case that the model has been changed to a ShiftedLinearModel.

new_model_y = J(x=1)

print(new_model_y)

# %%
# The new model can then be evaluated yielding the same result as above
new_model_y(y=2) 

# %%
# If we evaluate partially with the parameter y we also get a new model.
# However, because the model with respect to x was a regular model, the new
# model is also a regular model.

new_model_x = J(y=2)

print(new_model_x)

# %%
# The new model can then be evaluated yielding the same result as above
new_model_x(x=1)

# %%
# Three models
# ------------
#
# We can also define a sum model with three models

C = cuqi.model.Model(lambda z: z**3, 1, 1)

K = A + B + C

print(K)

#%%
# We can evaluate the model simply by calling it with specific inputs
# Notice the added shift value of 5*2 when partially evaluating
new_sumModel = K(y=2)

print(new_sumModel)

# %%
# Finally we can evaluate this model also
# We expect the result to be 1^2 + 5*2 + 3^3 = 38
new_sumModel(x=1, z=3)
# %%
