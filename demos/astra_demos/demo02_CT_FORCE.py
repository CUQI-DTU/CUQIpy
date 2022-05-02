#%% 
import sys
sys.path.append("../../") 
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import cuqi
from cuqi.astra.model import ShiftedFanBeam2DModel

#%%
# ===============================
# Load phantom
# ===============================

phantommat = spio.loadmat('data/PipePhantom')
phantom = phantommat['phantom']

fig = plt.figure()
plt.imshow(phantom, vmin=-0.2, vmax = 1.2); plt.colorbar(); plt.title("Phantom")

#%%
# ===============================
# CT Parameters
# ===============================
beam_type="fanflat_vec"
proj_type='line_fanflat'
det_count=50
angles=np.linspace(0, 2 * np.pi, 60) 
beamshift_x=-125.3
source_y=-600
detector_y=500 
det_length=411 
domain=(550, 550)

#%%
# ===============================
# Create synthetic data from phantom
# ===============================

# CT geometry for creating synthetic data from high resolution phantom
N = int(np.shape(phantom)[0])
A_fine = ShiftedFanBeam2DModel(beam_type=beam_type, 
                    proj_type=proj_type, 
                    im_size=(N, N), 
                    det_count=det_count, 
                    angles=angles, 
                    beamshift_x=beamshift_x, 
                    source_y=source_y, 
                    detector_y=detector_y, 
                    det_length=det_length, 
                    domain=domain)
m = det_count * len(angles)

# Forward projection
b_true = A_fine.forward(phantom.flatten())
fig = plt.figure()
A_fine.range_geometry.plot(b_true); plt.colorbar(); plt.title("clean sinogram")

#%% add noise
rnl = 0.02
e0 = np.random.normal(0, 1, np.shape(b_true))
noise_std = rnl*np.linalg.norm(b_true)/np.linalg.norm(e0)
b_data = b_true + noise_std*e0

fig = plt.figure()
A_fine.range_geometry.plot(b_data); plt.colorbar(); plt.title("noisy sinogram")

#%% ============================
# Inverse problem
# ============================

# Reconstruction geometry
N = 100
A = ShiftedFanBeam2DModel(beam_type=beam_type, 
                    proj_type=proj_type, 
                    im_size=(N, N), 
                    det_count=det_count, 
                    angles=angles, 
                    beamshift_x=beamshift_x, 
                    source_y=source_y, 
                    detector_y=detector_y, 
                    det_length=det_length, 
                    domain=domain)
m = det_count * len(angles)

# Plot back projection
fig = plt.figure()
A.domain_geometry.plot(A.adjoint(b_true)); plt.title("Back projection"); plt.colorbar()

# %% Setup likelihood
likelihood = cuqi.distribution.GaussianCov(mean = A, cov = noise_std**2).to_likelihood(b_data)

# ============================
# Gaussian prior
# ============================

# Setup prior
prior = cuqi.distribution.GaussianCov(mean = np.zeros(N**2), cov = 1/50)

# Setup sampler
x0 = np.zeros(N**2)
posterior = cuqi.distribution.Posterior(likelihood, prior)
sampler = cuqi.sampler.Linear_RTO(posterior,x0)

# Sample
samples = sampler.sample(N = 500, Nb = 100)

#%% Plot mean 
fig = plt.figure()
samples.plot_mean(vmin = -0.2, vmax = 1.2); plt.colorbar(); plt.title('Posterior mean')

#%% Plot std
fig = plt.figure()
samples.plot_std(); plt.colorbar(); plt.title('Posterior std')
#%% Plot chain
fig = plt.figure()
samples.plot_chain(5000); plt.title('Chain of pixel 5000')


#%% ============================
# Weighted Gaussian prior
# ============================

# Prior mask
mask = np.zeros((N,N))
radii = np.array([0.85, 0.35])
c = np.linspace(1,-1,N, endpoint=True)
[xx, yy] = np.meshgrid(c,c)
mask[xx**2 + yy**2 <= radii[0]**2] = 1
mask[xx**2 + yy**2 <= radii[1]**2] = 0

fig = plt.figure()
plt.imshow(mask); plt.colorbar(); plt.title('Prior mask')
#%%
# Setup prior
prior_mean = np.zeros(N**2)
prior_mean[mask.flatten()==1] = 0.5
prior_cov = 0.01*np.ones(N**2)
prior_cov[mask.flatten()==1] = 0.05
prior = cuqi.distribution.GaussianCov(mean = prior_mean, cov = prior_cov)

# Setup sampler
x0 = np.zeros(N**2)
posterior = cuqi.distribution.Posterior(likelihood, prior)
sampler = cuqi.sampler.Linear_RTO(posterior,x0)

#%% Sample
samples_WG = sampler.sample(N = 500, Nb = 100)

#%% Plot mean 
fig = plt.figure(figsize=(10,5))
fig.subplots_adjust(wspace=0.7)

ax = plt.subplot(131)
cs = ax.imshow(phantom, vmin=-0.2, vmax = 1.2)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.set_title("Phantom")

ax = plt.subplot(132)
cs = ax.imshow(np.mean(samples.samples,axis=-1).reshape(A.domain_geometry.shape), vmin=-0.2, vmax = 1.2)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.set_title("Post. mean - G prior")

ax = plt.subplot(133)
cs = ax.imshow(np.mean(samples_WG.samples,axis=-1).reshape(A.domain_geometry.shape), vmin=-0.2, vmax = 1.2)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.set_title("Post. mean - WG prior")

#%% Plot std
fig = plt.figure()
samples_WG.plot_std(); plt.colorbar(); plt.title('Posterior std')
#%% Plot chain
fig = plt.figure()
samples_WG.plot_chain(5000); plt.title('Chain of pixel 5000')
# %%
