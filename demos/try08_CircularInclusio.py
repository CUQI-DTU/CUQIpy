#%%
import sys
sys.path.append("../")
import time
import numpy as np
import matplotlib.pyplot as plt

# myfuns
import cuqi
from cuqi.sampler import pCN, MetropolisHastings
from cuqi.distribution import Gaussian, Posterior, DistributionGallery
from cuqi.samples import Samples
import dolfin as dl
mesh= dl.UnitSquareMesh(10,10)
Vh = dl.FunctionSpace(mesh,'CG',1)
geom = cuqi.fenicsGeometry.CircularInclusion(Vh)