import numpy as np
import pycuda.autoinit
import skcuda.cublas as cublas
import pycuda.gpuarray as gpuarray
import pycuda

import scipy.linalg
import scipy as sp

N = 100
N_BATCH = 1  # only 1 matrix to be decomposed
A_SHAPE = (N, N)

np.random.seed(42)

a = np.random.rand(*A_SHAPE).astype(np.float64)
a_gpu = gpuarray.to_gpu(a.T.copy())  # transpose a to follow "F" order

# print(a)
# print(a_gpu.ptr)

a_gpu_batch = np.asarray([a_gpu.ptr])
a_gpu_batch = gpuarray.to_gpu(a_gpu_batch)
p_gpu = gpuarray.zeros(N * N_BATCH, np.int32)

info = np.ones(N_BATCH).astype(np.int32) * -100
info_gpu = gpuarray.to_gpu(info)

cublas_handle = cublas.cublasCreate()
cublas.cublasDgetrfBatched(
    cublas_handle,
    N,
    a_gpu_batch.gpudata,
    N,
    p_gpu.gpudata,
    info_gpu.gpudata,
    N_BATCH,
)

cublas.cublasDestroy(cublas_handle)


l_cuda = np.tril(a_gpu.get().T, -1)
u_cuda = np.triu(a_gpu.get().T)

# reconstruct the permutation matrix A from the pivoting index
# https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-getrfbatched
p_cuda = np.eye(N)
for i, j in enumerate(p_gpu.get()):
    dia = np.eye(N)
    # swap
    dia[[i, j - 1]] = dia[[j - 1, i]]
    # dot product
    p_cuda = p_cuda @ dia

# get the tranpose of p_cuda, as p is applied to A in cuBLAS
# in the for P*A = L*U. Since the cuBLAS is column based, there
# is no need to perform transpose of the permutation matrix to find
# its inverse.
# p_cuda = p_cuda

# ref from scipy
p, l, u = sp.linalg.lu(a)

print(np.allclose(l, l_cuda + np.eye(N, dtype=np.float32)))
print(np.allclose(u, u_cuda))
print(np.allclose(p_cuda, p))
