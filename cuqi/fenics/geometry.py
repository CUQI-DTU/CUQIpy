from cuqi.geometry import Discrete, Geometry, MappedGeometry
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import ufl

class FEniCSContinuous(Geometry):

    def __init__(self, function_space, labels = ['x', 'y']):
        self.function_space = function_space
        if self.physical_dim >2:
            raise NotImplementedError("'FenicsContinuous' object does not support 3D meshes yet. 'mesh' needs to be a 1D or 2D mesh.")
        self.labels = labels

    @property
    def physical_dim(self):
        return self.function_space.mesh().geometry().dim()  

    @property
    def mesh(self):
        return self.function_space.mesh()

    @property
    def shape(self):
        return (self.function_space.dim(),)

    def par2fun(self,par):
        """The parameter to function map used to map parameters to function values in e.g. plotting."""
        par = self._process_values(par)
        Ns = par.shape[-1]
        fun_list = []
        for idx in range(Ns):
            fun = dl.Function(self.function_space)
            fun.vector().zero()
            fun.vector().set_local(par[...,idx])
            fun_list.append(fun)

        if len(fun_list) == 1:
            return fun_list[0]
        else:
            return fun_list

    def _plot(self,values,subplots=True,**kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        kwargs : keyword arguments
            keyword arguments which the function :meth:`dolfin.plot` normally takes.
        """
        if isinstance(values, dl.function.function.Function):
            Ns = 1
            values = [values]
        elif hasattr(values,'__len__'): 
            Ns = len(values)
        subplot_ids = self._create_subplot_list(Ns,subplots=subplots)

        ims = []
        for rows,cols,subplot_id in subplot_ids:
            fun = values[subplot_id-1]
            if subplots:
                plt.subplot(rows,cols,subplot_id); 
            ims.append(dl.plot(fun, **kwargs))

        self._plot_config(subplots) 
        return ims

    def _process_values(self, values):
        if isinstance(values, dl.function.function.Function):
            return [values]
        elif len(values.shape) == 1:
            values = values[..., np.newaxis]
        
        return values
    
    def _plot_config(self, subplot):
        if self.labels is not None:
            if subplot == False:
                plt.gca().set_xlabel(self.labels[0])
                if self.physical_dim == 2: plt.gca().set_ylabel(self.labels[1]) 
            else:
                for i, axis in enumerate(plt.gcf().axes):
                    axis.set_xlabel(self.labels[0])
                    if self.physical_dim == 2: axis.set_ylabel(self.labels[1])


class FEniCSMappedGeometry(MappedGeometry):
    """
    """
    def par2fun(self,p):
        funvals = self.geometry.par2fun(p)
        if isinstance(funvals, dl.function.function.Function):
            funvals = [funvals]
        mapped_value_list = []
        for idx in range(len(funvals)):
            mapped_value = self.map(funvals[idx]) 
            if isinstance(mapped_value, ufl.algebra.Operator):
                mapped_value_list.append(dl.project(mapped_value, self.geometry.function_space))
            elif isinstance(mapped_value,dl.function.function.Function):
                mapped_value_list.append(mapped_value)
            else:
                raise ValueError(f"'{self.__class__.__name__}.map' should return 'ufl.algebra.Operator'")
            
        if len(mapped_value_list) == 1:
            return mapped_value_list[0]
        else:
            return mapped_value_list
    
    def fun2par(self,f):
        raise NotImplementedError

#class CircularInclusion(Discrete, FenicsContinuous):
#
#    def __init__(self, function_space, inclusion_parameters=['radius','x','y'], labels = ['x', 'y']):
#        Discrete.__init__(self,inclusion_parameters)
#        FenicsContinuous.__init__(self,function_space,labels)
#        # assert len =3
#        if self.physical_dim !=2:
#            raise NotImplementedError("'CircularInclusion' object support 2D meshes only.")
#
#    @property
#    def shape(self):
#        #https://newbedev.com/calling-parent-class-init-with-multiple-inheritance-what-s-the-right-way
#        # super(Discrete,self).shape calls second parent shape
#        return super().shape #This calls first parent shape
#
#
#    def plot(self):
#        pass
#
#    def par2fun(self):
#        pass
