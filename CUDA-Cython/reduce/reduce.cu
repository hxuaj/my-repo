#include "error.cuh"
#include <stdio.h>
#include "reduce.hpp"

const int block_size = 128;


void __global__ reduce_kernel(const int N, float *v_dev, float *res_dev)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;

    __shared__ float s_y[128];
    s_y[tid] = (n < N) ? v_dev[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        res_dev[bid] = s_y[0];
    }
}

void reduce(int N, float *input, float *output)
{
    float *v_dev; // define device vector pointer
    const int M = sizeof(float) * N;
    CHECK(cudaMalloc(&v_dev, M));
    CHECK(cudaMemcpy(v_dev, input, M, cudaMemcpyHostToDevice));

    int grid_size = (N + block_size - 1) / block_size;
    // allocate memory for result vector in device
    const int res_mem = sizeof(float) * grid_size;
    float *res_dev;
    CHECK(cudaMalloc(&res_dev, res_mem));

    // kernel
    reduce_kernel<<<grid_size, block_size>>>(N, v_dev, res_dev);

    // allocate memory for result vector in host
    float *res_h = (float *) malloc(res_mem);
    CHECK(cudaMemcpy(res_h, res_dev, res_mem, cudaMemcpyDeviceToHost));

    float _temp = 0;
    for (int i = 0; i < grid_size; ++i)
    {
        _temp += res_h[i];
    }
    *output = _temp;

    CHECK(cudaFree(v_dev));
    CHECK(cudaFree(res_dev));
    free(res_h);
}