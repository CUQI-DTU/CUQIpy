from cuqi.geometry import Geometry
import numpy as np

class _ProductGeometry(Geometry):
    """ A class for representing a product geometry. A product geometry
    represents the product space of multiple geometries of type Geometry.
    See the example below for a product geometry of two geometries.

    Parameters
    ----------
    geometries : list of cuqi.geometry.Geometry
        List of geometries to be combined into a product geometry.

    Example
    -------
    .. code-block:: python
        import numpy as np
        from cuqi.geometry import Continuous1D, Discrete
        from cuqi.experimental.geometry import _ProductGeometry
        geometry1 = Continuous1D(np.linspace(0, 1, 100))
        geometry2 = Discrete(["sound_speed"])
        product_geometry = _ProductGeometry(geometry1, geometry2)
    """

    def __init__(self, *geometries):
        self.geometries = geometries

    @property
    def geometries(self):
        """List of geometries that are combined to form the product geometry."""
        return self._geometries
    
    @geometries.setter
    def geometries(self, geometries):
        self._geometries = geometries

    @property
    def fun_shape(self):
        """Shape of the function representation. Returns a tuple, where
        each element of the tuple is the shape of the function 
        representation of each geometry."""
        return tuple([g.fun_shape for g in self.geometries])
    
    @property
    def fun_dim(self):
        """Dimension of the function representation which is the sum of
        the function representation dimensions of each geometry."""
        return sum([g.fun_dim for g in self.geometries])

    @property
    def par_shape(self):
        """Shape of the parameter representation. Returns a tuple, where
        each element of the tuple is the shape of the parameter
        representation of each geometry."""
        return tuple([g.par_shape for g in self.geometries])
    
    @property
    def par_dim(self):
        """Dimension of the parameter representation which is the sum of
        the parameter representation dimensions of each geometry."""
        return sum(self.par_dim_list)
    
    @property
    def par_dim_list(self):
        """List of the parameter representation dimensions of each
        geometry. This property is useful for indexing a stacked parameter
        vector."""
        return [g.par_dim for g in self.geometries]
    
    @property
    def stacked_par_split_indices(self):
        """Indices at which the stacked parameter vector should be split
        to obtain the parameter vectors for each geometry. For example, if
        the stacked parameter vector is [1, 2, 3, 4, 5, 6] and the parameter
        vectors for each geometry are [1, 2], [3, 4], and [5, 6], then the
        split indices are [2, 4]"""
        return np.cumsum(self.par_dim_list[:-1])

    @property
    def number_of_geometries(self):
        """Number of geometries in the product geometry."""
        return len(self.geometries)
    
    def _split_par(self, par):
        """Splits a stacked parameter vector into parameter vectors for each
        geometry."""
        return tuple(np.split(par, self.stacked_par_indices))
    
    def _plot(self, values, **kwargs):
        """Plotting function for the product geometry."""
        raise NotImplementedError(
            f"Plotting not implemented for {self.__class__.__name__}.")

    def par2fun(self, *args):
        """Converts parameter vector(s) into function values for each 
        geometry. The parameter vector can be stacked (all parameters are
        in one vector) or unstacked (one parameter vector corresponds to
        each geometry)."""

        # If one argument is passed, then it is assumed that the parameter
        # vector is stacked and split it.
        # No effect if the parameter vector is already split and corresponds
        # to one geometry.
        if len(args) == 1:
            args = self._split_par(args[0])

        # Convert parameter vectors to function values for each geometry.
        fun_list = []
        for i, g in enumerate(self.geometries):
            fun_i = g.par2fun(args[i])
            fun_list.append(fun_i)
        return tuple(fun_list)

    def fun2par(self, *funvals, stacked=False):
        """Converts (multiple) function values into the corresponding
        parameter vectors. If the flag stacked is set to True, then the
        parameter vectors are stacked into one vector. Otherwise, the
        parameter vectors are returned as a tuple."""

        par_vec = []
        for i, g in enumerate(self.geometries):
            par_i = g.fun2par(funvals[i])
            par_vec.append(par_i)
        
        return np.hstack(par_vec) if stacked else tuple(par_vec)

    def vec2fun(self, *funvec):
        """Maps function vector representation, if available, to function 
        values."""
        fun_values = []  
        for i, g in enumerate(self.geometries):
            fun_values.append(g.vec2fun(funvec[i]))

        return tuple(fun_values)
    
    def fun2vec(self, fun):
        """Maps function values to a vector representation of the function values, if available."""
        vec_values = []
        for i, g in enumerate(self.geometries):
            vec_values.append(g.fun2vec(fun[i]))
        
        return tuple(vec_values)
        
    
    def __repr__(self) -> str:
        """Representation of the product geometry."""
        string = "{}(".format(self.__class__.__name__) + "of the following geometries:\n"
        for g in self.geometries:
            string += "\t{}\n".format(g.__repr__())
        string += ")"
        return string