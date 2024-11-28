"""
Algebra in CUQIpy
=================

CUQIpy provides a simple algebraic framework for defining and manipulating
variables.

In this example, we will demonstrate how to define a simple algebraic structure
and perform some basic algebraic operations.

"""

#%%
# Defining Variables
# ------------------
# To utilize the algebraic framework in CUQIpy, we first need to define some
# variables to apply the algebraic operations on. In this example, we will
# define variables `x` and `y`.

from cuqi.experimental.algebra import VariableNode

x = VariableNode('x')
y = VariableNode('y')

#%%
# Recording Algebraic Operations
# --------------------
# We can now perform some algebraic operations on the variables `x` and `y`.
# The algebraic operations are recorded in a computational graph (abstract syntax tree).
# The operations are recoding the correct ordering and adhering to the rules of
# algebra.

print("Basic operations:")
print(x + 1)
print(x + y)
print(x * y)
print(x / y)

print("\nComplex operations:")
print(x**2 + 2*x*y + y**2)
print((x + y)**2)

print("\nProgrammatric operations:")
print(x[2]+y[3])

# %%
# Utilizing the Computational Graph
# ---------------------------------
# The computational graph can be utilized to evaluate the algebraic expressions
# when desired. This means we can define mathematical expressions without
# needing to evaluate them immediately.

# Define a mathematical expression
expr1 = (x + y)**2

# Evaluate the expression (using the __call__ method)
print(f"Expression {expr1} evaluated at x=2, y=3 yields {expr1(x=2, y=3)}")

# Another example
expr2 = x**2 + 2*x*y + y**2 + 16
print(f"Expression {expr2} evaluated at x=2, y=3 yields {expr2(x=2, y=3)}")

# Another example utilizing array indexing
expr3 = x[1] + y[2]
print(f"Expression {expr3} evaluated at x=[1,2,3], y=[4,5,6] yields {expr3(x=[1,2,3], y=[4,5,6])}")
# %%
