try: 
    from . import pde
    from . import geometry
    from . import testproblem
    from . import model
    __all__ = [pde, geometry, testproblem, model]
    print("cuqi.fenics module is loaded.")

except ModuleNotFoundError as error:
    if error.name == "dolfin" or error.name == "ufl": 
        pass
    else:
        raise error