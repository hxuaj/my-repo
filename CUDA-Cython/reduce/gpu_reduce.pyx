"""
# A wrapper version witout using numpy. Don't include numpy.get_include() when compile.
cdef extern from "reduce.hpp":
    void reduce(int n, float *input, float *output)

def gpu_reduce(float[:] vector):
    cdef int n = vector.shape[0]
    cdef float res
    reduce(n, &vector[0], &res)

    return res

"""
# 在setup.py的include_dirs中加入np.get_include()然后build时报错：
# reduce.lib(tmpxft_0000459c_00000000-17_reduce.obj) : error LNK2038: 
# 检测到“RuntimeLibrary”的不匹配项: 值“MT_StaticRelease”不匹配值“MD_DynamicRelease”(gpu_reduce.obj 中)
# reduce.lib是通过nvcc编译.cu文件得到(对应静态C/C++ runtime library)，
# 而gpu_reduce.obj是动态编译的。
# fixed by add extra_compile_args '/MT', which causes the application
# to use the multithread, static version of the run-time library. 
# cudatoolkit 11.3.1, cython 0.29.28, numpy 1.21.5
import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "reduce.hpp":
    void reduce(int n, float *input, float *output)

def gpu_reduce(np.ndarray[np.float32_t, ndim=1] vector):
    cdef int n = vector.shape[0]
    cdef float res
    reduce(n, &vector[0], &res)

    return res