# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2020-06
# ========================================================================
import numpy as np
# from scipy.sparse import linalg as splinalg
import astra

#=========================================================================
# global: fixed parameters
#=========================================================================
N = 150           # object size N-by-N pixels
p = int(1.5*N)    # number of detector pixels
q = 90            # number of projection angles

# view angles
theta = np.linspace(0, 2*np.pi, q, endpoint=False)   # in rad

# problem setting
source_origin = 3*N                     # source origin distance [cm]
detector_origin = N                       # origin detector distance [cm]
detector_pixel_size = (source_origin+detector_origin)/source_origin
detector_length = detector_pixel_size*p   # detector length

# object dimensions
vol_geom = astra.create_vol_geom(N,N)

# object 
proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta, source_origin, detector_origin)
proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

# =================================================================
# forward model
# ================================================================= 
def A(x, flag):
    if flag == 1:
        # forward projection
        return proj_forward_sino(x)
    elif flag == 2:
        # backward projection  
         return proj_backward_sino(x)

#=========================================================================
def proj_forward_sino(x):     
    # forward projection
    _, Ax = astra.create_sino(x.reshape((N,N), order='F'), proj_id)
    Ax = np.fliplr(Ax)
    
    return Ax.flatten()

#=========================================================================
def proj_backward_sino(b):          
    # backward projection   
    b = np.fliplr(b.reshape((q, p)))
    _, ATb = astra.create_backprojection(b, proj_id)
    
    return ATb.flatten(order='F')