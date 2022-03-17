import pkg_resources
import scipy.io as spio

def satellite():
    stream = pkg_resources.resource_stream(__name__, 'data/satellite.mat')
    return spio.loadmat(stream)['x_true']