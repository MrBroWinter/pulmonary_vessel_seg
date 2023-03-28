
import numpy as np

a = np.ones((5,5,5))
b = np.repeat(a[None, ...],3, axis=0)
print(b.shape)