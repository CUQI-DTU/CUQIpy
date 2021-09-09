# %%
sys.path.append("..") 
import cuqi
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from PIL import Image

from cuqi.modelCT import CT
# %%
N = 32; n=N**2
p = 32
q = np.linspace(0,np.pi,30)
m = p*np.size(q)

x = io.loadmat("data/phantom512.mat")["X"]
x= np.array(Image.fromarray(x).resize((N,N)))
x = x.ravel()
model = CT(im_size=(N,N),det_count=p,angles=q)

#%%
prior = cuqi.distribution.Gaussian(np.zeros(n),1)
noise = cuqi.distribution.Gaussian(np.zeros(m),0.1)

b_exact = model@x
e = noise.sample().flatten()
b=b_exact+e

# %%
IP=cuqi.problem.Type1(b,model,noise,prior)
results=IP.sample(5000)
#%%
x_mean = np.mean(results.samples,axis=-1)
model.domain_geometry.plot(x_mean)
#%%
x_mean = np.std(results.samples,axis=-1)
model.domain_geometry.plot(x_mean)
# %%
