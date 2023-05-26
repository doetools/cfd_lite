import pycuda.gpuarray as gpuarray
import numpy as np
from typing import Tuple


class CFDCofficientsGPU:
    def __init__(self, grid_shape: Tuple[int, int]) -> None:
        self.ap_u = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.ap_v = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.ap_p = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.a_p = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.a_e = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.a_w = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.a_n = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.a_s = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.b = gpuarray.zeros(grid_shape, dtype=np.float32)

        self.flag_u = gpuarray.to_gpu(-1 * np.ones(grid_shape, dtype=np.int32))
        self.flag_v = gpuarray.to_gpu(-1 * np.ones(grid_shape, dtype=np.int32))
        self.flag_p = gpuarray.to_gpu(-1 * np.ones(grid_shape, dtype=np.int32))

        self.resistance_x = gpuarray.zeros(grid_shape, dtype=np.float32)
        self.resistance_y = gpuarray.zeros(grid_shape, dtype=np.float32)


if __name__ == "__main__":
    import pycuda.autoinit

    coeff = CFDCofficientsGPU((10, 10))
    print(coeff.flag_p)
