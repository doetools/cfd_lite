import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import skcuda.linalg as culinalg

culinalg.init()

np.random.seed(42)

N = 3
block = (128, 1, 1)
grid = (int((N - 0.1) / 128) + 1, 1, 1)
# print(grid)

a = np.random.randn(N, N).astype(np.float32)
# make a diagonally dominant
for i in range(N):
    a[i][i] = np.abs(a[i]).sum() - abs(a[i][i])

b = np.random.randn(N).astype(np.float32)

b_gpu = gpuarray.to_gpu(b)
a_gpu = gpuarray.to_gpu(a)

a_inv_gpu = culinalg.inv(a_gpu)
a_inv = a_inv_gpu.get()
# print(a_inv)
# print(a_inv @ a)


x_gpu = culinalg.dot(a_inv_gpu, b_gpu)
x = x_gpu.get()
# print(x)

print(a @ x - b)
print(a_inv @ b)
print(x)

# x_np = np.linalg.solve(a, b)
# print(x_np)
