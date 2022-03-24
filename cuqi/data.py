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

def rgb2gray(img):
    return img @ np.array([0.2125, 0.7154, 0.0721])

def imresize(image, size, **kwargs):
    """Resize an image to a new size.
    
    kwargs are passed to scipy.ndimage.zoom.
    """
    zoom_factor = size/np.array(image.shape)
    return zoom(image, zoom_factor, **kwargs)
