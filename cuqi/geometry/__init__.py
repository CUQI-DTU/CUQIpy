from ._geometry import (
    Geometry,
    _WrappedGeometry,
    Continuous,
    Continuous1D,
    Continuous2D,
    Image2D,
    Discrete,
    MappedGeometry,
    _DefaultGeometry,
    _DefaultGeometry1D,
    _DefaultGeometry2D,
    KLExpansion,
    KLExpansion_Full,
    CustomKL,
    StepExpansion
)


# TODO: We will remove the use of identity geometries in the future
_identity_geometries = [_DefaultGeometry1D, _DefaultGeometry2D, Continuous1D, Continuous2D, Discrete, Image2D]

def _get_identity_geometries():
    """Return the geometries that have identity `par2fun` and `fun2par` methods (including those where `par2fun` and `fun2par` perform reshaping of the parameters or the function values array. e.g. the geometry `Image2D`.).
    These geometries do not alter the gradient computations.
    """
    return _identity_geometries
