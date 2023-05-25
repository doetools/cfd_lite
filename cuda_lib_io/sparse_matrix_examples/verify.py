import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.io as io
import scipy.linalg as linalg


# read mtx
a = io.mmread("example.mtx")
N, M = a.todense().shape
b = np.ones(N, dtype=np.float32)
x = spsolve(a, b)
x_read = np.loadtxt(
    "/home/sesa461392/Documents/wei_dev/parallelize_inverse/sparse_matrix/GLU_public-master/src/x.dat",
    dtype=np.float32,
)
print(x_read - x)
print(np.allclose(x, x_read, rtol=0.001, atol=0.001))
