import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit

x_size = 10000000
x = np.random.randn(x_size).astype(dtype=np.float32)
x_gpu = gpuarray.to_gpu(x)

with open("./cfd_cuda_lib/utility.cu", "r") as f:
    source = f.read()

kernels = SourceModule(source)
reduce = kernels.get_function("reduce")

block_size = 32 * 32
block = (block_size, 1, 1)
num_block = (
    int(x_size / block_size)
    if x_size % block_size == 0
    else int(x_size / block_size) + 1
)
grid = (num_block, 1, 1)

y = np.zeros(num_block, dtype=np.float32)
y_gpu = gpuarray.to_gpu(y)

reduce(
    x_gpu.gpudata,
    np.int32(x_size),
    y_gpu.gpudata,
    np.int32(num_block),
    block=block,
    grid=grid,
)

sum = y_gpu.get().sum()
print(sum, x.sum())
