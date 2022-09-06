# ===================================================================
# Created by:
# Felipe Uribe @ DTU
# ===================================================================
# Version 2021-04
# ===================================================================
import numpy as np
import scipy as sp
import scipy.stats as sps
import warnings



# ===================================================================
# ===================================================================
# ===================================================================
def Geweke(X, A = 0.1, B = 0.5):
    # Geweke's MCMC convergence diagnostic:
    # Test for equality of the means of the first A% (default 10%)  
    # and last B (50%) of a Markov chain
    Ns, _ = X.shape
    nA = int(np.floor(A*Ns))
    nB = int(Ns - np.floor(B*Ns))
    
    # extract sub chains
    X_A, X_B = X[:nA, :], X[nB:, :]
    
    # compute the mean
    mean_X_A, mean_X_B = X_A.mean(axis=0), X_B.mean(axis=0)

    # Spectral estimates for variance
    var_X_A, var_X_B = spectrum0(X_A), spectrum0(X_B)

    # params of the geweke
    z = (mean_X_A - mean_X_B) / (np.sqrt((var_X_A/nA) + (var_X_B/(Ns-nB+1))))
    p = 2*(1-sps.norm.cdf(abs(z)))
    
    # show
    idx1 = np.where(p >= 0.95)   
    idx2 = np.where(p < 0.95)    
    warnings.warn("Geweke's diagnostics is a work-in-progress")
    print('Geweke test passed at indices ', idx1, '\n') 
    print('Geweke test NOT passed at indices ', idx2, '\n')
        
    return z, p

# ===================================================================
def spectrum0(x):
    # Spectral density at frequency zero
    # Marko Laine <marko.laine@fmi.fi>
    m, n = x.shape
    s = np.empty(n)
    for i in range(n):
        # check this later: sp.signal.welch(x[:, i])[0] to avoid 'spectrum' fun
        spec, _ = spectrum(x[:, i], m) 
        s[i] = spec[0]

    return s

# ===================================================================
def spectrum(x, nfft):
    # Power spectral density using Hanning window
    # Marko Laine <marko.laine@fmi.fi>
    n = len(x)
    nw = int(np.fix(nfft/4))
    noverlap = int(np.fix(nw/2))
    if (n < nw):
        x[nw], n = 0, nw
        
    # Hanning window
    idx = np.arange(1, nw+1, 1)
    w = 0.5*(1 - np.cos(2*np.pi*idx/(nw+1))) # check this later: np.hanning(nw)
    
    # estimate PSD
    k = int(np.fix((n-noverlap)/(nw-noverlap)))    # number of windows
    kmu = k*np.linalg.norm(w)**2                   # normalizing scale factor
    y = np.zeros(nfft)
    for _ in range(k):
        xw = w*x[idx-1]
        idx += (nw - noverlap)
        Xx = abs(np.fft.fft(xw, nfft))**2
        y += Xx
    y = y*(1/kmu) # normalize
    
    n2 = int(np.floor(nfft/2))
    idx2 = np.arange(0, n2, 1)
    y = y[idx2]
    f = 1/(n*idx2)

    return y, f



# ===================================================================
# ===================================================================
# ===================================================================