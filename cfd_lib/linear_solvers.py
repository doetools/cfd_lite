from cfd_lib.solver_io import CFDCofficients
from typing import Union, List, Callable
import numpy as np


def gauss_seidel(
    cfd_coefficients: CFDCofficients,
    x: Union[np.array, List[float]],
    sor=1.0,
    num_i=0,
    num_j=0,
    num_iterations=300,
    skip_eval=lambda i, j: False,
) -> None:

    a_e = cfd_coefficients.a_e
    a_w = cfd_coefficients.a_w
    a_n = cfd_coefficients.a_n
    a_s = cfd_coefficients.a_s
    b = cfd_coefficients.b
    a_p = cfd_coefficients.a_p

    index = 0

    while index < num_iterations:
        for i in range(1, num_i):
            for j in range(1, num_j):

                if skip_eval(i, j):
                    continue

                x_new = (
                    a_e[i, j] * x[i, j + 1]
                    + a_w[i, j] * x[i, j - 1]
                    + a_n[i, j] * x[i + 1, j]
                    + a_s[i, j] * x[i - 1, j]
                    + b[i, j]
                ) / a_p[i, j]

                x[i, j] = sor * x_new + (1 - sor) * x[i, j]

        index += 1
