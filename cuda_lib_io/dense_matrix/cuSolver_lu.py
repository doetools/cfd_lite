import numpy as np
import scipy.linalg
import scipy as sp
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver

np.random.seed(42)

h = solver.cusolverDnCreate()

N = 3
X_SHAPE = (N, N)
x = np.random.rand(*X_SHAPE).astype(np.float32)

# Need to copy transposed matrix because T only returns a view:
m, n = x.shape
x_gpu = gpuarray.to_gpu(x.T.copy())

# Set up work buffers:
Lwork = solver.cusolverDnSgetrf_bufferSize(h, m, n, x_gpu.gpudata, m)
workspace_gpu = gpuarray.zeros(Lwork, np.float32)
devipiv_gpu = gpuarray.zeros(min(m, n), np.int32)
devinfo_gpu = gpuarray.zeros(1, np.int32)

# Compute:
solver.cusolverDnSgetrf(
    h,
    m,
    n,
    x_gpu.gpudata,
    m,
    workspace_gpu.gpudata,
    devipiv_gpu.gpudata,
    devinfo_gpu.gpudata,
)

# Confirm that solution is correct by checking against result obtained with
# scipy; set dimensions of computed lower/upper triangular matrices to facilitate
# comparison if the original matrix was not square:
l_cuda = np.tril(x_gpu.get().T, -1)
u_cuda = np.triu(x_gpu.get().T)

p, l, u = sp.linalg.lu(x)

print(x_gpu.get().T)
print(l)
print(u)
print(p)
print(devipiv_gpu.get().T)

print("lower triangular matrix is correct: %r" % np.allclose(np.tril(l, -1), l_cuda))
print("upper triangular matrix is correct: %r" % np.allclose(np.triu(u), u_cuda))


solver.cusolverDnDestroy(h)
