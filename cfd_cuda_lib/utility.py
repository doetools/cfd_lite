import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import cfd_cuda_lib.cu as cu
from cfd_cuda_lib.build import build_kenerls
from cfd_cuda_lib.solver_io import CFDCofficientsGPU
from typing import Tuple
from pytools import memoize


def sum_vector(x_gpu: gpuarray) -> np.float32:

    reduce = build_kenerls().get_function("reduce")

    x_size = x_gpu.size

    num_block = (
        int(x_size / cu.N_BLOCK_2D)
        if x_size % cu.N_BLOCK_2D == 0
        else int(x_size / cu.N_BLOCK_2D) + 1
    )

    grid = (num_block, 1, 1)

    y_gpu = gpuarray.zeros(num_block, dtype=np.float32)

    reduce(
        x_gpu.gpudata,
        np.int32(x_size),
        y_gpu.gpudata,
        np.int32(num_block),
        block=cu.BLOCK_1D,
        grid=grid,
    )

    if num_block < 10:
        return y_gpu.get().sum()
    else:
        return sum_vector(y_gpu)


def gauss_seidel(
    coeff: CFDCofficientsGPU,
    x: gpuarray,
    x_type: gpuarray,
    sor=1.0,
    num_iterations=300,
) -> None:
    # variables
    x_size = x.size
    _, row_size = x.shape

    # colors
    colors_gpu = get_gs_colors(x_size, row_size)
    # print(colors_gpu.get().reshape(x.shape))
    # print(x_type.get().reshape(x.shape))

    # get gs kernel
    gauss_seidel_cu = build_kenerls().get_function("gauss_seidel")
    grid = get_1d_grid(x_size)

    # run gs
    count = 0
    while count < num_iterations:
        # 1 inner iteration
        for target_color in [cu.RED, cu.GREEN]:
            gauss_seidel_cu(
                x,
                coeff.a_e,
                coeff.a_w,
                coeff.a_n,
                coeff.a_s,
                coeff.a_p,
                coeff.b,
                np.float32(sor),
                x_type,
                colors_gpu,
                np.int32(target_color),
                np.int32(x_size),
                np.int32(row_size),
                block=cu.BLOCK_1D,
                grid=grid,
            )

        count += 1


def get_1d_grid(size: int, block_size=cu.N_BLOCK_2D) -> Tuple[int, int, int]:
    num_block = get_1d_num_block(size, block_size)
    return (num_block, 1, 1)


def get_1d_num_block(size: int, block_size=cu.N_BLOCK_2D) -> int:
    num_block = int(size / block_size)
    if size % block_size == 0:
        return num_block
    else:
        return num_block + 1


@memoize
def get_gs_colors(x_size: int, row_size: int) -> gpuarray:
    gs_colors_cu = build_kenerls().get_function("gs_colors")
    colors_gpu = gpuarray.zeros(x_size, dtype=np.int32)
    grid = get_1d_grid(x_size)
    gs_colors_cu(colors_gpu.gpudata, np.int32(row_size), block=cu.BLOCK_1D, grid=grid)

    return colors_gpu


if __name__ == "__main__":

    np.random.seed(42)
    x_size = 1024 * 100
    x = np.random.randn(x_size).astype(dtype=np.float32)

    x_gpu = gpuarray.to_gpu(x)
    total = sum_vector(x_gpu)

    print(total, x.sum())
