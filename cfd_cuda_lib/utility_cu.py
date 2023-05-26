import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import cu
from build import build_kenerls


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


if __name__ == "__main__":

    np.random.seed(42)
    x_size = 1024 * 100
    x = np.random.randn(x_size).astype(dtype=np.float32)

    x_gpu = gpuarray.to_gpu(x)
    total = sum_vector(x_gpu)

    print(total, x.sum())
