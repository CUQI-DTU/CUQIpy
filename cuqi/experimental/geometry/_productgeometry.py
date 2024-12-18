from cuqi.geometry import Geometry

class ConcatenatedGeometries(Geometry):
    """A class that represents a concatenation of geometries.

    Parameters
    ----------
    geometries : list of cuqi.geometry.Geometry
        List of geometries to concatenate.
    """

    def __init__(self, *geometries):
        self.geometries = geometries

    @property
    def geometries(self):
        """List of geometries in the concatenation."""
        return self._geometries
    
    @geometries.setter
    def geometries(self, geometries):
        self._geometries = geometries

    @property
    def fun_shape(self):
        """Shape of the function space. Each element of the tuple corresponds 
        to the shape of the function space of each geometry."""
        return tuple([g.fun_shape for g in self.geometries])
    
    @property
    def fun_dim(self):
        """Dimension of the function space which is the sum of the function
        dimensions of each geometry."""
        return sum([g.fun_dim for g in self.geometries])

    @property
    def par_shape(self):
        """Shape of the parameter space. Each element of the tuple corresponds
        to the shape of the parameter space of each geometry."""
        return tuple([g.par_shape for g in self.geometries])
    
    @property
    def par_dim(self):
        """Dimension of the parameter space which is the sum of the parameter
        dimensions of each geometry."""
        return sum(self.par_dim_list)
    
    @property
    def par_dim_list(self):
        """List of the parameter dimensions of each geometry."""
        return [g.par_dim for g in self.geometries]
    
    @property
    def stack_indices(self):
        """Indecies of the parameter vector where each geometry's parameter
        vector starts and ends. For example, if the parameter vector is
        [p1, p2, p3] and the parameter dimensions of the geometries are
        [2, 3, 4], then the stack_indecies will be [0, 2, 5, 9]."""
        return np.cumsum([0] + self.par_dim_list)

    @property
    def number_of_geometries(self):
        """Number of geometries in the concatenation."""
        return len(self.geometries)
    
    def _plot(self, values, **kwargs):
        """Plotting function for concatenated geometries. """
        pass

    def par2fun(self, pars):
        """Converts a parameter vector into (multiple) function values each
        corresponding to a geometry in the concatenation."""

        # Compute the cumulative sum of the parameter dimensions of each
        # geometry to simplify indexing.
        par_dim_cumsum = np.cumsum(self.par_dim_list)

        # append 0 to the beginning of the array to account for the first
        # geometry.
        par_dim_cumsum = np.insert(par_dim_cumsum, 0, 0)

        # Compute the function values for each geometry and return them as a
        # a list.
        fun_list = []
        for i, g in enumerate(self.geometries):
            par_i = pars[par_dim_cumsum[i]:par_dim_cumsum[i+1]]
            fun_i = g.par2fun(par_i)
            fun_list.append(fun_i)
        return fun_list

    def fun2par(self, *funvals):
        """Converts (multiple) function values into a parameter vector."""
        # Create parameter vector
        pars = np.empty(self.par_dim)

        # Compute the cumulative sum of the parameter dimensions of each
        # geometry to simplify indexing of the parameter vector.
        par_dim_cumsum = np.cumsum(self.par_dim_list)

        # append 0 to the beginning of the array to account for the first
        # geometry.
        par_dim_cumsum = np.insert(par_dim_cumsum, 0, 0)
        
        # Fill parameter vector with the parameter vectors of each geometry.
        for i, g in enumerate(self.geometries):
            fun_i = funvals[i]
            par_i = g.fun2par(fun_i)
            print(par_i.shape)
            pars[par_dim_cumsum[i]:par_dim_cumsum[i+1]] = par_i

        return pars

    def vec2fun(self, funvec):
        """Maps function vector representation, if available, to function 
        values."""    
        raise NotImplementedError("vec2fun not implemented. ")
    
    def fun2vec(self, fun):
        """Maps function values to a vector representation of the function values, if available."""
        raise NotImplementedError("fun2vec not implemented. ")
    
    def __repr__(self) -> str:
        """Representation of the concatenated geometries."""
        string = "{}(".format(self.__class__.__name__) + "of the following geometries:\n"
        for g in self.geometries:
            string += "\t{}\n".format(g.__repr__())
        string += ")"
        return string