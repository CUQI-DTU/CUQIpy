from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Geometry(ABC):

    @abstractmethod
    def plot(self,input):
        pass

class Continous1D(Geometry):

    def __init__(self,dim):
        if len(dim)==1:
            self.grid = np.arange(dim[0])
        else:
            raise NotImplemented("Cannot init 1D geometry with spatial dimension > 1")

    def plot(self,input,*args,**kwargs):
        return plt.plot(self.grid,input,*args,**kwargs)