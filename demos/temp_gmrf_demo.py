
#%%
import cuqi
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
mean = np.zeros(50)
g1 = cuqi.distribution.GMRF(mean, 1, order=1, bc_type='neumann')
g1_mat = g1._prec_op.get_matrix().copy()
print(g1_mat)
plt.subplot(1,2,1)
g1.sample(10).plot()

import cuqi
import numpy as np
np.random.seed(0)
g2 = cuqi.distribution.GMRF(mean, 1, order=1, bc_type='some_bc')
g2_mat = g2._prec_op.get_matrix().todense()
print(g2_mat)
plt.subplot(1,2,2)
g2.sample(10).plot()