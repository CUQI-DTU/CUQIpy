Class structure
===============

Classes should always inherit from the `BaseClass` class of the module.

Classes should start with an __init__ method that sets up the class.
After this comes any public methods, followed by any public properties.
Finally private methods or attributes should be defined last.

Example of defining a specific distribution in the distribution module.

Note: Distribution requires a _logd and _sample method defined at the minimum.

.. code-block:: python

    import numpy as np
    from cuqi.distribution import Distribution
    from cuqi.utilities import force_ndarray

    class Normal(Distribution):

        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def _logd(self, x):
            return 1/(self.std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-self.mean)/self.std)**2)

        def _sample(self, n):
            return np.random.normal(self.mean, self.std, n)

        @property
        def mean(self):
            return self._mean
        
        @mean.setter
        def mean(self, value):
            self._mean = force_ndarray(value) #This ensures that mean is a numpy array.     


On testing
===========  

Testing is done using the `pytest` module. Tests are defined in the `tests` 
folder. Tests should be implemented for newly added features and for bug fixes.

Some resources and comments on testing:

* Introductory material on code testing from the CodeRefinery project: https://coderefinery.github.io/testing/
* We advise using test fixtures for setup code ("arrange" code) https://docs.pytest.org/en/6.2.x/fixture.html
