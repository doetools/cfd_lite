# OpenCL Algorithms to Solve AX=B

## Goals

Develop a suite of OpenCL kernels to compute A\*X=B by performing a sparse LU decomposition, when A is a sparse matrix.

Test if the inverse-based method is faster than iterative method,like Jacobi and Gauss Seidel solver.

## Implementation

- It is not viable to just copy the algorithms for dense matrix. The reasons are two-foled: first, the storage requirement for matrix A will be too big for a GPU (~1000 G for a million-cell CFD solve); second, the L and U factors may not be sparse and thus the storage for them will be impractically big.
- Use the sparse LU decomposition for factorizing sparse matrix A, like scipy.sparse.linalg.splu, which calls the [SuperLU](https://portal.nersc.gov/project/sparse/superlu/), which will make L and U factors sparse, too.

### General notes

1. [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html): determine the format to save the sparse matrix A. Using the np.array format naively won't work.
2. [dgetrf](https://netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f_source.html): get the LU decomposition of A as A = P \* L \* U. Directly finding the inverse is practically impossible.
3. [dgetrs](https://netlib.org/lapack/explore-html/d6/d49/dgetrs_8f_source.html): effctively find inverse of A. This is effectively solve P \* L \* U \* inv(A) = I, where inv(A) is the inverse of A, and I is a unity matrix.
4. determine X by using X = inv(A) \* B.

### CSC and MTX format

CSC stands for Compressed Sparse Column format, which is used in the University of Florida matrices.

### Main building blocks

1. matrix multiplication
2. matrix transpose
3. matrix row swap (interchange)
4. LU decomposition (with pivoting)
5. triangular matrix inverse

## Resources

### CUDA

1. [scikit-cuda](https://scikit-cuda.readthedocs.io/en/latest/index.html)
2. [cuBLAS](https://docs.nvidia.com/cuda/cublas/)
3. [cuSolver](https://docs.nvidia.com/cuda/cusolver/)

### OPENCL

1. [clmath](https://github.com/clMathLibraries)
2. [amd-sdk-fix](https://github.com/clockfort/amd-app-sdk-fixes/tree/master/samples/opencl/cl/app)
