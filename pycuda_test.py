#!/bin/python3
"""
These steps have been copied from the PyCUDA tutorial site:
  https://documen.tician.de/pycuda/tutorial.html
"""
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Create a 3D array of given dtype and each dimension of tile_sz * f
dtype = np.int32
tile_sz = 10
f = 120
K_X = f * tile_sz
K_Y = f * tile_sz
K_Z = f * tile_sz
a = np.zeros((K_X, K_Y, K_Z), dtype=dtype)

# The cuda code simply gets global thread number and sets the corresponding
# index in the array to that.
mod = SourceModule(f"""
  __global__ void test_kernel(int* a)
  {{

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

  a[idx] = idx;
  }}
  """)
func = mod.get_function("test_kernel")

# use PyCUDA to execute kernel on the numpy array
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

block=(tile_sz, tile_sz, tile_sz)
grid=(K_X//tile_sz, K_Y//tile_sz, K_Z//tile_sz)

func(a_gpu, block=block, grid=grid)
cuda.Context.synchronize()

a_result = np.empty_like(a)
cuda.memcpy_dtoh(a_result, a_gpu)
cuda.Context.synchronize()
a_gpu.free()

a_cpu = np.arange(K_X * K_Y * K_Z, dtype=dtype).reshape((K_X, K_Y, K_Z))
assert(K_X * K_Y * K_Z == np.max(a_result, axis=(0, 1, 2)) + 1)
assert((a_result == a_cpu).all())

print(f"PASSED: (arr_sz: {a.nbytes/(1024*1024*1024)} GB)")

