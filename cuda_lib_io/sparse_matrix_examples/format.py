import numpy as np
import scipy.sparse as sparse
import scipy.io as io

N = 10

a = sparse.random(N, N, format="csc", density=0.1, dtype=np.float32) + sparse.eye(
    N, dtype=np.float32
)

a_csc = sparse.csc_matrix(a.todense())
a_csr = sparse.csr_matrix(a.todense())


io.mmwrite(
    "a_csc.mtx",
    a_csc,
    comment="",
    field="real",
)

io.mmwrite(
    "a_csr.mtx",
    a_csr,
    comment="",
    field="real",
)
