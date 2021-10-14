from cuqi.geometry import Discrete, Geometry
import numpy as np
import matplotlib.pyplot as plt
import warnings

try: 
    import dolfin as dl
except Exception as error:
    warnings.warn(error.msg)


class FenicsContinuous(Geometry):

    def __init__(self, Vh, labels = ['x', 'y']):
        self.Vh = Vh
        if self.physical_dim >2:
            raise NotImplementedError("'FenicsContinuous' object does not support 3D meshes yet. 'mesh' needs to be a 1D or 2D mesh.")
        self.labels = labels

    @property
    def physical_dim(self):
        return self.Vh.mesh().geometry().dim()  

    @property
    def mesh(self):
        return self.Vh.mesh()

    @property
    def shape(self):
        return self.Vh.dim()


    def plot(self,values,subplots=True,**kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        kwargs : keyword arguments
            keyword arguments which the function :meth:`dolfin.plot` normally takes.
        """
        values = self._process_values(values)
        subplot_ids = self._create_subplot_list(values,subplots=subplots)

        ims = []
        func = dl.Function(self.Vh)
        for rows,cols,subplot_id in subplot_ids:
            func.vector().zero()
            func.vector().set_local(values[...,subplot_id-1])
            if subplots:
                plt.subplot(rows,cols,subplot_id); 
            ims.append(dl.plot(func, **kwargs))

        self._plot_config(subplots) 
        return ims

    def _process_values(self, values):
        if len(values.shape) == 1:
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


class CircularInclusion(Discrete, FenicsContinuous):

    def __init__(self, Vh, inclusion_parameters=['radius','x','y'], labels = ['x', 'y']):
        Discrete.__init__(self,inclusion_parameters)
        FenicsContinuous.__init__(self,Vh,labels)
        # assert len =3
        if self.physical_dim !=2:
            raise NotImplementedError("'CircularInclusion' object support 2D meshes only.")

    @property
    def shape(self):
        #https://newbedev.com/calling-parent-class-init-with-multiple-inheritance-what-s-the-right-way
        # super(Discrete,self).shape calls second parent shape
        return super().shape #This calls first parent shape


    def plot(self):
        pass

    def par2fun(self):
        pass