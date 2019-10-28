#!/usr/bin/env python3
"""
These steps have been copied from the PyCUDA tutorial site:
    https://documen.tician.de/pycuda/tutorial.html

A simple test if PyCUDA is working correctly on a single GPU.
"""

import numpy as np
import pycuda.autoinit # pylint: disable=unused-import
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Create a 3D array of given dtype and each dimension of tile_sz * f
DTYPE = np.int32
TILE_SZ = 10
F = 115
K_X = F * TILE_SZ
K_Y = F * TILE_SZ
K_Z = F * TILE_SZ
a = np.ones((K_X, K_Y, K_Z), dtype=DTYPE)

# This function simply gets the global thread number and indexes into the given
# array based on that, adding the thread number to the value there.
func = SourceModule("""
__global__ void test_kernel(int* a) {

    int threadsPerBlock  = blockDim.x * blockDim.y * blockDim.z;
    int threadNumInBlock = threadIdx.x +
                             (threadIdx.y * blockDim.x) +
                             (threadIdx.z * blockDim.x * blockDim.y);
    int blockNumInGrid   = blockIdx.x +
                             (blockIdx.y * gridDim.x) +
                             (blockIdx.z * gridDim.x * gridDim.y);
    int globalThreadNum  = threadNumInBlock +
                             (blockNumInGrid * threadsPerBlock);
    int idx = globalThreadNum;

    a[idx] += idx;
}""").get_function("test_kernel")

print(f"array size: {a.nbytes/(1024*1024*1024)} GB")
print("Allocating space on the GPU and copying the input array there")
a_gpu = cuda.mem_alloc(a.nbytes) # pylint: disable=no-member
cuda.memcpy_htod(a_gpu, a) # pylint: disable=no-member

# Call the CUDA kernel with a total of K_X * K_Y * K_Z threads
print("Running test CUDA kernel")
block=(TILE_SZ, TILE_SZ, TILE_SZ)
grid=(K_X//TILE_SZ, K_Y//TILE_SZ, K_Z//TILE_SZ)
func(a_gpu, block=block, grid=grid)
cuda.Context.synchronize() # pylint: disable=no-member

print("Copying result to CPU from GPU")
a_result = np.empty_like(a)
cuda.memcpy_dtoh(a_result, a_gpu) # pylint: disable=no-member
a_gpu.free()

print("Comparing result to CPU reference")
a_cpu = np.arange(K_X * K_Y * K_Z, dtype=DTYPE).reshape((K_X, K_Y, K_Z)) + a
assert K_X * K_Y * K_Z == np.max(a_result, axis=(0, 1, 2))
assert(a_result == a_cpu).all()

print("PASSED!")
