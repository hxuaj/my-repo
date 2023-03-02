from gpu_reduce import gpu_reduce
import numpy as np


a = np.array([1., 2., 3., 4.], dtype=np.float32)
sum = gpu_reduce(a)
print(sum)