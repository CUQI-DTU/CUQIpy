"""
Defining and using Joint Models
===============================

CUQIpy supports the definition of joint models. Joint models are models that
take multiple inputs and produce a single output. Joint models are defined as
the sum of two or more models. The following example shows how to define a
joint model with two inputs and one output and how to evaluate the model.


"""

# %%
# First we import the necessary modules
import cuqi

# %%
# Defining the joint model
# ------------------------
# 
# Suppose we have two models A: x -> z and B: y -> z
# In this case we consider a regular Model and a LinearModel

A = cuqi.model.Model(lambda x: x**2, 1, 1)
B = cuqi.model.LinearModel(lambda y: 5*y, lambda z: 5*z, 1, 1)

# %%
# The joint linear model J: (x, y) -> z is then defined
# as the sum of the two models A and B, i.e. J(x, y) = A(x) + B(y)
# In CUQIpy this is done by passing the two models to the JointModel class

J = cuqi.model.JointModel([A, B])

print(J)

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
