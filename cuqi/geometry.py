from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Geometry(ABC):

    @abstractmethod
    def plot(self,input):
        pass

class Continuous1D(Geometry):

    def __init__(self,dim,grid=None):
        if len(dim)==1:
            if grid is None:
                self.grid = np.arange(dim[0])
            else:
                self.grid = grid
        else:
            raise NotImplemented("Cannot init 1D geometry with spatial dimension > 1")

    def plot(self,values,*args,**kwargs):
        return plt.plot(self.grid,values,*args,**kwargs)