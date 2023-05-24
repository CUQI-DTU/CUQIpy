import cuqi

class RandomVariable:
    """ Random variable class """

    def __init__(self, distribution, operations=None, op_print=None, name=None):
        self._distribution = distribution
        if operations is None:
            self._operations = [] # list of operations to apply to the random variable
        else:
            self._operations = operations
        self._op_print = op_print
        self._name = name

    @property
    def _non_default_args(self):
        return [self._distribution.name]

    def __call__(self, *args, **kwargs):
        """ Apply operations to random variable """
        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError("Cannot pass both positional and keyword arguments to RandomVariable")

        if len(args) > 1:
            raise ValueError("Cannot pass more than one positional argument to RandomVariable")
        
        if len(kwargs) > 1:
            raise ValueError("Cannot pass more than one keyword argument to RandomVariable")
        
        if len(args) > 0:
            input = args[0]
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if key in self._non_default_args:
                    input = value

        # Call operations on the input
        for operation in self._operations:
            input = operation(input)
        return input

    def __repr__(self):
        out = "RandomVariable\n"
        out += "Original: "+str(self.name)+" ~ " + self._distribution.__repr__() + "\n"
        out += "Transformations: " + self.op_print
        return out
    
    @property
    def name(self):
        if self._name is None:
            self._name = cuqi.utilities._get_python_variable_name(self)
        return self._name
    
    # Printing of operations on random variable
    @property
    def op_print(self) -> str:
        if self._op_print is None:
            if self.name is None:
                self._op_print = ""
            else:
                self._op_print = self.name
        return self._op_print   
    
    @op_print.setter
    def op_print(self, value):
        self._op_print = value

    def __truediv__(self, other):
        """ Divide random variable by something """
        # Copy list of operations
        operations = self._operations.copy()
        operations.append(lambda x: x / other)
        op_print = self.op_print + " / " + str(other)
        return RandomVariable(self._distribution, operations, op_print, self.name)
    
    def __rtruediv__(self, other):
        """ Divide random variable by something """
        # Copy list of operations
        operations = self._operations.copy()
        operations.append(lambda x: other / x)
        op_print = str(other) + " / (" + self.op_print + ")"
        return RandomVariable(self._distribution, operations, op_print, self.name)






    
