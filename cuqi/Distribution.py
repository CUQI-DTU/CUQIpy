#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.special import  erf

class Normal(object):
    
    def __init__(self, mean, std, dims):
        self.mean = mean
        self.std = std
        self.dims = dims
    
    def sample(self):
        
        return np.random.normal(self.mean, self.std,self.dims)
    
    def pdf(self,x):
        return 1/(self.std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-self.mean)/self.std)**2)
    
    def cdf(self,x):
        return 0.5*(1 + erf((x-self.mean)/(self.std*np.sqrt(2))))
    
    def logpdf(self,x):
        return -np.log(self.std*np.sqrt(2*np.pi))-0.5*((x-self.mean)/self.std)**2