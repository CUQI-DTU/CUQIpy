#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt

from cuqi.ProbDist.Distribution import Normal

mean = -2.0
std = 4.0
dims = (5, 1)

pX = Normal(mean, std, dims)

s = pX.sample()
print(s)

xx = np.linspace(mean-3*std, mean+3*std, num=1001)
yy = pX.pdf(xx)

plt.figure()
plt.plot(xx, pX.pdf(xx))
plt.plot(xx, pX.cdf(xx))

plt.figure()
plt.plot(xx, pX.logpdf(xx))
plt.show()