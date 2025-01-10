from cuqi.geometry import Geometry
import numpy as np

class _ProductGeometry(Geometry):
    """ A class for representing a product geometry. A product geometry
    represents the product space of multiple geometries of type :class:`Geometry`.
    See the example below for a product geometry of two geometries.

    Parameters
    ----------
    \*geometries : cuqi.geometry.Geometry
        The geometries to be combined into a product geometry. Each geometry
        is passed as a comma-separated argument.

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
        # Check if all geometries are of type Geometry.
        for g in geometries:
            if not isinstance(g, Geometry):
                raise TypeError(
                    "All geometries must be of type Geometry. "
                    "Received: {}".format(type(g))
                )
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
        return tuple(np.split(par, self.stacked_par_split_indices))
    
    def _plot(self, values, **kwargs):
        """Plotting function for the product geometry."""
        raise NotImplementedError(
            f"Plotting not implemented for {self.__class__.__name__}.")

    def par2fun(self, *pars):
        """Converts parameter vector(s) into function values for each 
        geometry. The parameter vector can be stacked (all parameters are
        in one vector) or unstacked (one parameter vector corresponds to
        each geometry). In all cases, the order of the parameter vectors
        should follow the order of the geometries in the product, i.e., the
        first parameter vector corresponds to the first geometry and so on."""

        # If one argument is passed, then it is assumed that the parameter
        # vector is stacked and split it.
        # No effect if the parameter vector is already split and corresponds
        # to one geometry.
        if len(pars) == 1:
            pars = self._split_par(pars[0])

        # Convert parameter vectors to function values for each geometry.
        funvals = []
        for i, g in enumerate(self.geometries):
            funval_i = g.par2fun(pars[i])
            funvals.append(funval_i)
        return tuple(funvals)

    def fun2par(self, *funvals, stacked=False):
        """Converts (multiple) function values into the corresponding
        parameter vectors. If the flag stacked is set to True, then the
        parameter vectors are stacked into one vector. Otherwise, the
        parameter vectors are returned as a tuple. The order of function
        values should follow the order of the geometries in the product,
        i.e., the first function value corresponds to the first geometry
        and so on."""

        pars = []
        for i, g in enumerate(self.geometries):
            par_i = g.fun2par(funvals[i])
            pars.append(par_i)

        # stack parameters:
        if stacked:
            # if single sample
            if len(pars[0].shape) == 1:
                stacked_val = np.hstack(pars)
            elif len(pars[0].shape) == 2:
                stacked_val = np.vstack(pars)
            else:
                raise ValueError(
                    "Cannot stack parameter vectors with more than 2 dimensions."
                    )

        return  stacked_val if stacked else tuple(pars)

    def vec2fun(self, *funvecs):
        """Maps function vector representation, if available, to function 
        values. The order of the function vectors should follow the order of
        the geometries in the product, i.e., the first function vector
        corresponds to the first geometry and so on."""
        funvals = []  
        for i, g in enumerate(self.geometries):
            funvals.append(g.vec2fun(funvecs[i]))

        return tuple(funvals)
    
    def fun2vec(self, *funvals):
        """Maps function values to a vector representation of the function
        values, if available. The order of the function values should follow
        the order of the geometries in the product, i.e., the first function
        value corresponds to the first geometry and so on."""
        funvecs = []
        for i, g in enumerate(self.geometries):
            funvecs.append(g.fun2vec(funvals[i]))
        
        return tuple(funvecs)
        
    
    def __repr__(self) -> str:
        """Representation of the product geometry."""
        string = "{}(".format(self.__class__.__name__) + "\n"
        for g in self.geometries:
            string += "\t{}\n".format(g.__repr__())
        string += ")"
        return string