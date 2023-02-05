"""
Joint Linear Model
==================

CUQIpy supports the definition of joint linear models. The following example
shows how to define a joint linear model with two inputs and one output.


"""

# %%
import cuqi

# Suppose we have two models A: x -> z and B: y -> z
A = cuqi.model.LinearModel(lambda x: x, lambda z: z, 1, 1)
B = cuqi.model.LinearModel(lambda y: y, lambda z: z, 1, 1)

# Suppose we want to define a joint linear model J: (x, y) -> z
# as the sum of the two models A and B, i.e. J(x, y) = A(x) + B(y)
J = cuqi.model.JointModel([A, B], type="sum") # type sum is default

# We can inspect the model
print(J)

# %%
# We can evaluate the model simply by calling it with specific inputs
# In this case both A and B are identity functions, so J(x, y) = x + y
J(x=1, y=2)

# %%
# We can also partially evaluate the model
# This returns a new model that is a function of the remaining input
# in this case J(x=1) gives a new model taking y -> 1 + y
new_model = J(x=1)

print(new_model)

# %%
# The new model can then be evaluated
#TODO: this should be 3 (shift parameter needs to be added to Model class evaluation)
new_model(y=2) 

 # %%
