from cfd_lib.cfd_constants import NON_FLUID
import numpy as np
from typing import Tuple
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import numpy.random as random

random.seed(42)


class CFDCofficientsTest:
    def __init__(self, grid_shape: Tuple[int, int]) -> None:
        self.a_p = np.zeros(grid_shape, dtype=np.float32)
        self.a_e = np.zeros(grid_shape, dtype=np.float32)
        self.a_w = np.zeros(grid_shape, dtype=np.float32)
        self.a_n = np.zeros(grid_shape, dtype=np.float32)
        self.a_s = np.zeros(grid_shape, dtype=np.float32)
        self.b = np.zeros(grid_shape, dtype=np.float32)


def for_each(shape):
    I, J = shape

    for i in range(1, I - 1):
        for j in range(1, J - 1):
            yield (i, j)


def generate_coefficients(coeff: CFDCofficientsTest, shape):
    size = np.prod(shape)

    # get x
    x = np.zeros(SHAPE, dtype=np.float32)

    # get x_type
    x_type = -1 * np.ones(SHAPE, dtype=np.int32)
    x_type[:, 0], x_type[:, -1], x_type[0, :], x_type[-1, :] = (
        NON_FLUID,
        NON_FLUID,
        NON_FLUID,
        NON_FLUID,
    )

    # get b
    coeff.b = random.rand(size).reshape(shape).astype(np.float32)

    for (i, j) in for_each(SHAPE):
        coeff.a_e[i, j] = random.normal(size=1)[0]
        coeff.a_w[i, j] = random.normal(size=1)[0]
        coeff.a_n[i, j] = random.normal(size=1)[0]
        coeff.a_s[i, j] = random.normal(size=1)[0]
        coeff.a_p[i, j] = (
            abs(coeff.a_e[i, j])
            + abs(coeff.a_w[i, j])
            + abs(coeff.a_n[i, j])
            + abs(coeff.a_s[i, j])
        )

    return x, x_type, coeff


if __name__ == "__main__":

    from cfd_lib.linear_solvers import gauss_seidel
    import cfd_cuda_lib.utility as utility

    SHAPE = (10, 10)
    coeff = CFDCofficientsTest(SHAPE)

    # generate coeff
    x, x_type, coeff = generate_coefficients(coeff, SHAPE)

    coeff_gpu = CFDCofficientsTest(SHAPE)
    coeff_gpu.a_e = gpuarray.to_gpu(coeff.a_e)
    coeff_gpu.a_w = gpuarray.to_gpu(coeff.a_w)
    coeff_gpu.a_n = gpuarray.to_gpu(coeff.a_n)
    coeff_gpu.a_s = gpuarray.to_gpu(coeff.a_s)
    coeff_gpu.a_p = gpuarray.to_gpu(coeff.a_p)
    coeff_gpu.b = gpuarray.to_gpu(coeff.b)
    x_gpu = gpuarray.to_gpu(x)
    x_type_gpu = gpuarray.to_gpu(x_type)

    # # cpu solve
    # num_i, num_j = SHAPE
    # gauss_seidel(
    #     coeff,
    #     x,
    #     sor=1.0,
    #     num_i=num_i - 1,
    #     num_j=num_j - 1,
    #     num_iterations=300,
    #     skip_eval=lambda i, j: x_type[i, j] == NON_FLUID,
    # )
    # print(x)

    # # gpu solve
    # utility.gauss_seidel(coeff_gpu, x_gpu, x_type_gpu, sor=1.0, num_iterations=300)
    # print(x_gpu.get())

    # if np.allclose(x, x_gpu.get(), atol=1e-5):
    #     print(f"True")
    # else:
    #     diff = np.abs(x - x_gpu.get())
    #     print(f"{diff.max()}")

    utility.gauss_seidel(coeff_gpu, x_gpu, x_type_gpu, sor=1.0, num_iterations=300)
    a = x_gpu.get().copy()
    print(a)

    x_gpu = gpuarray.to_gpu(x)
    utility.gauss_seidel(coeff_gpu, x_gpu, x_type_gpu, sor=1.15, num_iterations=300)
    b = x_gpu.get().copy()

    diff = np.abs(a - b)
    print(b)
    print(diff.max())
