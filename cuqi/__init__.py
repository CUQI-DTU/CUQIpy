from . import data
from . import diagnostics
from . import distribution
from . import geometry
from . import likelihood
from . import model
from . import operator
from . import pde
from . import problem
from . import sampler
from . import samples
from . import solver
from . import testproblem
from . import utilities

# ==== Import engines ====
# Silently fails on import cuqi,
# but gives error on import cuqi.engine

engines = []

# FEniCS
try:
    from . import fenics
    engines.append("fenics")
except ModuleNotFoundError:
    pass

# ASTRA-TOOLBOX
try:
    from . import astra
    engines.append("astra")
except ModuleNotFoundError:
    pass

#Print extra modules

if len(engines)>0:
    print("3rd party engines available for cuqi:",end=" ")
    print(*engines)

del engines