from hello import test
import time
import numpy as np

def py_test(n):

    s = 0
    for i in range(n):
        for j in range(n):
            s += i * j
    
    return s

cython_time = []
cython_res = []
python_time = []
python_res = []
eps = 1e-10

for i in range(10, 15):

    n = 2 ** i
    tic = time.time()
    cython_res.append(test(n))
    toc = time.time()
    cython_time.append(toc- tic)

    tic = time.time()
    python_res.append(py_test(n))
    toc = time.time()
    python_time.append(toc- tic)

print("cython_time: {}".format(cython_time))
print("python_time: {}".format(python_time))
print("cython_res: {}".format(cython_res))
print("python_res: {}".format(python_res))
print("if cython_res == python_res ---> {}".format(cython_res == python_res))
print(np.array(python_time) / (np.array(cython_time) + eps))