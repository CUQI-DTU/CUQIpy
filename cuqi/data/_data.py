import pkg_resources
import scipy.io as spio
import numpy as np
import scipy.ndimage as spnd
import matplotlib.pyplot as plt

def satellite(size=None):
    """Photograph of a satelite."""
    stream = pkg_resources.resource_stream(__name__, 'satellite.mat')
    img= spio.loadmat(stream)['x_true']
    if size:
        img = imresize(img, size) 
    return img

def astronaut(size=None, grayscale=True):
    """Color image of the astronaut Eileen Collins.

    See https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut
    """
    stream = pkg_resources.resource_stream(__name__, 'astronaut.npz')
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
    stream = pkg_resources.resource_stream(__name__, 'camera.npz')
    img = np.load(stream)['arr_0']
    if size:
        img = imresize(img, size) 
    return img

def cat(size=None, grayscale=True):
    """Chelsea the cat.

    See https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.cat
    """
    stream = pkg_resources.resource_stream(__name__, 'cat.npz')
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

# Python translation of matlab code by Felipe Uribe @ DTU Compute 2020. 
def shepp_logan(size=128):
    """Modified Shepp-Logan phantom.
    
    This head phantom is the same as the Shepp-Logan except the intensities
    are changed to yield higher contrast in the image.

    Parameters
    ----------
    size : int
        Size of the image to generate. Image is square with sides of length size.

    Returns
    -------
    img : ndarray
        Image of the phantom.

    Notes
    -----
    Python translation from phantomgallery code in AIRToolsII
    https://github.com/jakobsj/AIRToolsII.

    Original paper:
    L. A. Shepp and B. F. Logan, “The Fourier reconstruction of a head section,”
    in IEEE Transactions on Nuclear Science, vol. 21, no. 3, pp. 21-43, June 1974.
    DOI:10.1109/TNS.1974.6499235
        
    """
    # image is N x N
    N = size

    #                  A      a      b     x0      y0    phi
    e = np.array( [ [  1,    .69,   .92,    0,       0,   0 ], 
                    [-.8,  .6624, .8740,    0,  -.0184,   0 ],
                    [-.2,  .1100, .3100,  .22,       0,  -18],
                    [-.2,  .1600, .4100, -.22,       0,   18],
                    [ .1,  .2100, .2500,    0,     .35,   0 ],
                    [ .1,  .0460, .0460,    0,      .1,   0 ],
                    [ .1,  .0460, .0460,    0,     -.1,   0 ],
                    [ .1,  .0460, .0230, -.08,   -.605,   0 ],
                    [ .1,  .0230, .0230,    0,   -.606,   0 ],
                    [ .1,  .0230, .0460,  .06,   -.605,   0 ] ] )
    
    xn = ((np.arange(0,N)-(N-1)/2) / ((N-1)/2))
    Xn = np.tile(xn, (N,1))
    Yn = np.rot90(Xn)
    X  = np.zeros((N,N))

    # for each ellipse to be added     
    nn = len(e)
    for i in range(nn):
        A   = e[i,0]
        a2  = e[i,1]**2
        b2  = e[i,2]**2
        x0  = e[i,3]
        y0  = e[i,4]
        phi = e[i,5]*np.pi/180        
        #
        x   = Xn-x0
        y   = Yn-y0
        idd = ((x*np.cos(phi) + y*np.sin(phi))**2)/a2 + ((y*np.cos(phi) - x*np.sin(phi))**2)/b2
        idx = np.where( idd <= 1 )

        # add the amplitude of the ellipse
        X[idx] += A
    
    idx    = np.where( X < 0 )
    X[idx] = 0

    return X

# Python translation of matlab code by Felipe Uribe @ DTU Compute 2020. 
def threephases(size=128, p=70):
    """Three-phase phantom.
    
    Creates an image with three different phases.

    Parameters
    ----------
    size : int
        Size of the image to generate. Image is square with sides of length size.

    p : int
        Number of blobs in each phase (blobs can overlap).

    Returns
    -------
    img : ndarray
        Image of the phantom.

    Notes
    -----
    Python translation from phantomgallery code in AIRToolsII
    https://github.com/jakobsj/AIRToolsII.
    
    """

    # image is N x N
    N = size

    # 1st
    xx     = np.arange(1,N+1)-1
    I, J   = np.meshgrid(xx,xx, indexing='xy')
    sigma1 = 0.025*N
    c1     = np.random.rand(p,2)*N
    x1     = np.zeros((N,N))
    for i in range(p):
        x1 += np.exp(-abs(I-c1[i,0])**3/(2.5*sigma1)**3 - abs(J-c1[i,1])**3/sigma1**3)
    t1 = 0.35
    x1[x1 < t1]  = 0
    x1[x1 >= t1] = 2

    # 2nd
    sigma2 = 0.03*N
    c2     = np.random.rand(p,2)*N
    x2     = np.zeros((N,N))
    for i in range(p):
        x2 += np.exp(-(I-c2[i,0])**2/(2*sigma2)**2 - (J-c2[i,1])**2/sigma2**2)
    t2 = 0.55
    x2[x2 < t2]  = 0
    x2[x2 >= t2] = 1

    # combine the two images
    x = x1 + x2
    x[x == 3] = 1
    x = x/x.max()
    
    return x

# Python translation of matlab code by Felipe Uribe @ DTU Compute 2020. 
def p_power(size=128, relnz=0.3, p=2, seed=1): #relnz=0.65, p=2.3
    """ p-power class phantom.
    
    Create an image generated from a random pattern of nonzero pixels
    with correlation between pixels controlled by p and sparsity by relnz.

    Note: image will change when varying size. To avoid this change image
    size using :meth:`cuqi.data.imresize` after generating the image.

    Parameters
    ----------
    size : int
        Size of the image to generate. Image is square with sides of length size.
    
    relnz : float
        Relative number of nonzero pixels.

    p : int
        Power of the pattern. Structure (correlation) increases with larger p.

    seed : int
        Seed for the random number generator.

    Returns
    -------
    ndarray
        Image of the phantom.   

    Notes
    -----
    Python translation from phantomgallery code in AIRToolsII
    https://github.com/jakobsj/AIRToolsII.

    Original paper:
    Jorgensen, Jakob S., et al. "Empirical average-case relation between undersampling
    and sparsity in x-ray CT." Inverse problems and imaging (Springfield, Mo.) 9.2 (2015): 431.

    """

    # Set random seed
    rng = np.random.RandomState(seed)

    # image is N x N
    N = size

    # check if image size is odd
    if N/2 == round(N/2):
        Nodd = False
    else: 
        Nodd = True
        N += 1
    
    # Random pixels
    P   = rng.randn(N,N)

    # Create the pattern
    xx   = np.arange(1,N+1)
    I, J = np.meshgrid(xx,xx, indexing='xy')
    U = ( ( (2*I-1)/N - 1)**2 + ( (2*J-1)/N - 1)**2 )**(-p/2)
    F = U*np.exp(2*np.pi*np.sqrt(-1+0j)*P)
    F = abs(np.fft.ifft2(F))
    f = -np.sort(-F.flatten(order='F'))   # 'descend'
    k = round(relnz*N**2)-1
    F[F < f[k]] = 0
    x = F/f[0]
    if Nodd:
        x = F[1:-1,1:-1]
    
    return x

def cookie(size=128, grayscale=True):
    """ Cartoon-style image of a cookie.

    The image is generated from the cookie.png color image file in cuqi/data.
    The original image is of size 2491 x 2243 pixels. The image is resized
    to be square and optionally converted to grayscale.

    Parameters
    ----------
    size : int
        Size of the image to generate. Image is square with sides of length size.

    grayscale : bool
        If True, return grayscale image. Otherwise return RGB image.
        Small values in the grayscale image are set to zero to make the
        background completely black.

    Returns
    -------
    ndarray
        Image of the phantom.

    """

    # Read cookie.png file and convert to rgb
    stream = pkg_resources.resource_stream(__name__, 'cookie.png')
    cookie = plt.imread(stream)

    # Convert to rgb
    cookie = cookie[..., :3]

    # Convert to grayscale
    if grayscale:
        cookie = rgb2gray(cookie)
        
    # Resize
    cookie = imresize(cookie, size)

    if grayscale:
        cookie[cookie < 0.05] = 0 # Make background completely black

    return cookie

def rgb2gray(img):
    """ Convert RGB image to grayscale using the colorimetric (luminosity-preserving) method
    
    See e.g. discussion in https://poynton.ca/PDFs/ColorFAQ.pdf page 6 on the benefit of this
    method compared to the classical [0.299, 0.587, 0.114] weights.
    
    """
    return img @ np.array([0.2125, 0.7154, 0.0721])

def imresize(image, size, **kwargs):
    """Resize an image to a new size.
    
    kwargs are passed to scipy.ndimage.zoom.
    """
    zoom_factor = size/np.array(image.shape)
    return spnd.zoom(image, zoom_factor, **kwargs)
