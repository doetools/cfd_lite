#%%
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.io as io
import scipy.linalg as linalg

N = 10000

## generate a sparse maptrix
a = sparse.random(N, N, format="csr", density=0.00001, dtype=np.float32) + sparse.eye(
    N, dtype=np.float32
)

# write mtx
io.mmwrite(
    "example.mtx",
    a,
    comment="",
    field="real",
)
