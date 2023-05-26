from utility_cu import get_gs_colors
import numpy as np
import pycuda.autoinit

x = np.zeros((10, 10))
x_size = x.size
_, row_size = x.shape
colors_gpu = get_gs_colors(x_size, row_size)

print(colors_gpu.get().reshape(x.shape))
