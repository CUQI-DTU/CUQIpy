#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt

import cuqi

mean = -2.0
std = 4.0

pX = cuqi.distribution.Normal(mean, std)

s = pX.sample()
print(s)

#Plot pdf, cdf and logpdf
xx = np.linspace(mean-3*std, mean+3*std, num=1001)
yy = pX.pdf(xx)

plt.figure()
plt.plot(xx, pX.pdf(xx))
plt.title('pdf')

plt.figure()
plt.plot(xx, pX.cdf(xx))
plt.title('cdf')

plt.figure()
plt.plot(xx, pX.logpdf(xx))
plt.title('logpdf')
