import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culinalg

import numpy as np

import time


def iterative_method(a: np.array, b: np.array, outer_num=1) -> np.array:
    block = (128, 1, 1)
    grid = (int((N - 0.1) / 128) + 1, 1, 1)

    x = np.random.randn(N).astype(np.float32)

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

    # emulate the cfd solver for species
    while outer_num > 0:
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

        outer_num -= 1

    x = x_gpu.get()

    return x


def inverse_method(a: np.array, b: np.array, outer_num=1) -> np.array:

    culinalg.init()

    b_gpu = gpuarray.to_gpu(b)
    a_gpu = gpuarray.to_gpu(a)

    a_inv_gpu = culinalg.inv(a_gpu)

    # emulate the cfd solver for species
    while outer_num > 0:
        x_gpu = culinalg.dot(a_inv_gpu, b_gpu)
        outer_num -= 1

    x = x_gpu.get()

    return x


if __name__ == "__main__":
    np.random.seed(42)

    N = 8000
    OUTER_NUM = 1000
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    # make a diagonally dominant
    for i in range(N):
        a[i][i] = np.abs(a[i]).sum() - abs(a[i][i])

    start = time.time()
    x_jacobi = iterative_method(a, b, outer_num=OUTER_NUM)
    jacobi_time = time.time() - start

    start = time.time()
    x_direct = inverse_method(a, b, outer_num=OUTER_NUM)
    inverse_time = time.time() - start

    # print(x_direct, x_jacobi)

    print(np.allclose(x_direct, x_jacobi))
    print(f"jacobi method is {jacobi_time}, inverse method is {inverse_time}")
