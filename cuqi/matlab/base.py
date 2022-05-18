import matlab.engine
import matlab
import numpy as np
import os

class MatlabInstance:
    """ Matlab instance allowing calls to Matlab functions.

    The functions are called using the MatlabInstance.__call__ method.
    
    Example
    -------
    .. code-block:: python

        import cuqi
        import numpy as np
        matlab_instance = cuqi.matlab.MatlabInstance()
        x = np.ones(10)
        y = np.ones(10)*2
        z = matlab_instance("plus", x, y) # The string is the matlab function to call
        assert np.array_equal(z, x+y)

    For more examples see demos/matlab_demos/call_function.py.

    Notes
    -----
    On creation a Matlab engine is started with working directory set to the
    current working directory. On destruction the engine is automatically closed.

    """
    def __init__(self):
        self.eng = matlab.engine.start_matlab()

    def __del__(self):
        self.eng.quit()    

    def __repr__(self):
        # Get current folder
        folder = self.eng.pwd()

        # Get all .m files in folder
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.m')]

        # Display current folder and all .m files found
        return f'Matlab instance. \n Directory: {folder}\n Found .m files: {files}'

    def __call__(self, func, *args, nargout=1):
        """ Calls Matlab function with given arguments.
        
        Arguments
        ---------
        func : str
            Matlab function name (must be in Matlab's path)

        *args
            Arguments to pass to Matlab function.
            Numpy arrays are efficiently converted to Matlab arrays.

        nargout : int
            Number of output arguments the Matlab function returns.

        Returns
        -------
        result from Matlab function
        Matlab arrays are efficiently converted to numpy arrays.

        """
        # Check if func ends with .m, if so remove it
        if func.endswith('.m'):
            func = func[:-2]

        # convert to numpy arrays to matlab arrays
        args_matlab = [matlab.double(x) if isinstance(x, np.ndarray) else x for x in args]

        # call matlab function with matlab array
        out_matlab = getattr(self.eng, func)(*args_matlab, nargout=nargout)

        # convert output to numpy arrays
        if nargout == 1:
            if isinstance(out_matlab, matlab.double):
                return np.array(out_matlab).squeeze()
            else:
                return out_matlab
        else:
            out = (np.array(x).squeeze() if isinstance(x, matlab.double) else x for x in out_matlab)
            return tuple(out)
