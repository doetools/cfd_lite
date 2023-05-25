import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import numpy as np

np.random.seed(42)

N = 3
block = (128, 1, 1)
grid = (int((N - 0.1) / 128) + 1, 1, 1)
# print(grid)

a = np.random.randn(N, N).astype(np.float32)
x = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)


# make a diagonally dominant
for i in range(N):
    a[i][i] = np.abs(a[i]).sum() - abs(a[i][i])

# print(a)

a_gpu = gpuarray.to_gpu(a)
x_gpu = gpuarray.to_gpu(x)
b_gpu = gpuarray.to_gpu(b)
tmp_gpu = gpuarray.zeros(N, dtype=np.float32)


mod = SourceModule(
    """
    __global__ void jacobi(float *a, float *x, float *b, float *tmp, int N, float sor)
    {
      int idx = threadIdx.x + blockIdx.x*blockDim.x;
      int i;
      float sum = 0.0;

      for (i=0; i<N; i++){
        if (i != idx)
            sum += a[idx*N+i] * x[i];
      }

      tmp[idx] = sor * (b[idx] - sum)/ a[idx*N + idx] + (1 - sor) * x[idx];
    }
    """
)

func = mod.get_function("jacobi")

num_iterations = 100

while num_iterations > 0:
    func(
        a_gpu.gpudata,
        x_gpu.gpudata,
        b_gpu.gpudata,
        tmp_gpu.gpudata,
        np.int32(N),
        np.float32(1.0),
        block=block,
        grid=grid,
    )

    # swap
    x_gpu = tmp_gpu.copy()

    num_iterations -= 1


x_new = x_gpu.get()
print(x_new)
print(a @ x_new - b)


# # ref
# x_np = np.linalg.solve(a, b)
# # print(x_np)
# print(a @ x_np - b)

# # check closeness
# print(np.allclose(x_new, x_np))
