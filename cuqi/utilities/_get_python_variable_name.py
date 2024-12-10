import sys
from itertools import count
import re

import warnings
import cuqi


def _get_python_variable_name(var):
    """ Retrieve the Python variable name of an object. Takes the first variable name appearing on the stack that is not in the ignore list. """

    ignored_var_names = ["self", "cls", "obj", "var", "_", "result", "args", "kwargs", "par_name", "name", "distribution", "dist"]

    # First get the stack size and loop (in reverse) through the stack
    # It can be a bit slow to loop through stack size so we limit the levels
    stack_size = _stack_size2a() 

    for i in reversed(range(min(stack_size, cuqi.config.MAX_STACK_SEARCH_DEPTH))): 

        # For each frame we look for a variable name matching the object (var)
        local_vars = sys._getframe(i).f_locals.items()
        var_names = [var_name for var_name, var_val in local_vars if var_val is var]

        # If we find a matching variable name we return it if it is not in the ignore list
        if len(var_names) > 0:
            # Get variable names not regex matching the ignore list
            var_names = [var_name for var_name in var_names if not any([re.match(ignore_var_name, var_name) for ignore_var_name in ignored_var_names])]
            
            if len(var_names) > 0:
                return var_names[0]

    warnings.warn("Could not automatically find variable name for object. Did you assign (=) the object to a python variable? Alternatively, use keyword `name` when defining distribution to specify a name. If code runs slowly and variable name is not needed set config.MAX_STACK_SEARCH_DEPTH to 0. These names are reserved {} and should not be used as object name.".format(ignored_var_names))

    return None

#https://stackoverflow.com/questions/34115298/how-do-i-get-the-current-depth-of-the-python-interpreter-stack
def _stack_size2a(size=2):
    """Get stack size for caller's frame.
    """
    frame = sys._getframe(size)

    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size
