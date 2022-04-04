import pkg_resources
import scipy.io as spio
import numpy as np
from scipy.ndimage import zoom

def satellite(size=None):
    """Photograph of a satelite."""
    stream = pkg_resources.resource_stream(__name__, 'data/satellite.mat')
    img= spio.loadmat(stream)['x_true']
    if size:
        img = imresize(img, size) 
    return img

def astronaut(size=None, grayscale=True):
    """Color image of the astronaut Eileen Collins.

    See https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut
    """
    stream = pkg_resources.resource_stream(__name__, 'data/astronaut.npz')
    img = np.load(stream)['arr_0']
    if grayscale:
        img = rgb2gray(img)
    if size:
        img = imresize(img, size) 
    return img

def camera(size=None):
    """Camera man.

    See https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.camera
    """
    stream = pkg_resources.resource_stream(__name__, 'data/camera.npz')
    img = np.load(stream)['arr_0']
    if size:
        img = imresize(img, size) 
    return img

def cat(size=None, grayscale=True):
    """Chelsea the cat.

    See https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.cat
    """
    stream = pkg_resources.resource_stream(__name__, 'data/cat.npz')
    img = np.load(stream)['arr_0']
    if grayscale:
        img = rgb2gray(img)
    if size:
        img = imresize(img, size) 
    return img

# Python translation of matlab code by Felipe Uribe @ DTU Compute 2020. 
def grains(size=128, num_grains=34, seed=1):
    """ Generate a random image of grains built from Voronoi cells.
    
    Parameters
    ----------
    size : int
        Size of the image to generate. Image is square with sides of length size.

    num_grains : int
        Number of grains in the image.

    seed : int
        Random seed for grain structure.
    
    Returns
    -------
    img : ndarray
        Image of grains.

    Notes
    -----
    Python translation from phantomgallery code in AIRToolsII
    https://github.com/jakobsj/AIRToolsII.
    """

    # Grains image must have more than two grains
    if num_grains <= 2:
        raise ValueError("num_grains must be greater than 2")

    # Set the random seed
    rng = np.random.RandomState(seed)

    # Image size is N x N
    N         = size
    dN        = round(N/10)
    Nbig      = N + 2*dN
    total_dim = Nbig**2

    # random pixels whose coordinates (xG,yG,zG) are the "centre" of the grains
    xG = np.ceil(Nbig*rng.rand(num_grains, 1))
    yG = np.ceil(Nbig*rng.rand(num_grains, 1))

    # set up voxel coordinates for distance computation
    xx   = np.arange(1, Nbig+1)
    X, Y = np.meshgrid(xx, xx, indexing='xy')
    X    = X.flatten(order='F')
    Y    = Y.flatten(order='F')

    # for centre pixel k [xG(k),yG(k),zG(k)] compute the distance to all the 
    # voxels in the box and store the distances in column k.
    distArray = np.zeros((total_dim, num_grains))
    for k in range(num_grains):
        distArray[:,k] = (X-xG[k])**2 + (Y-yG[k])**2

    # determine to which grain each of the voxels belong. This is found as the
    # centre with minimal distance to the given voxel
    minIdx = np.argmin(distArray, axis=1)

    # reshape to 2D, subtract 1 to have 0 as minimal value, extract the
    # middle part of the image, and scale to have 1 as maximum value
    img   = minIdx.reshape(Nbig, Nbig) - 1
    img   = img[np.ix_(dN+np.arange(1, N+1)-1, dN+np.arange(1, N+1)-1)]
    img   = img/img.max()
    
    return img

def rgb2gray(img):
    return img @ np.array([0.2125, 0.7154, 0.0721])

def imresize(image, size, **kwargs):
    """Resize an image to a new size.
    
    kwargs are passed to scipy.ndimage.zoom.
    """
    zoom_factor = size/np.array(image.shape)
    return zoom(image, zoom_factor, **kwargs)
