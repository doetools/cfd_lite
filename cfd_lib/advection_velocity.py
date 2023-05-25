from cfd_lib.solver_io import SolverIO
from typing import Tuple


def get_face_v(u, v, i, j, velocity="u") -> Tuple[float, float, float, float]:
    """
    get face velocity on the control volume of u/v in staggered grid
    """
    if velocity == "u":
        Fe = 0.5 * (u[i, j + 1] + u[i, j])
        Fw = 0.5 * (u[i, j - 1] + u[i, j])
        Fn = 0.5 * (v[i, j] + v[i, j + 1])
        Fs = 0.5 * (v[i - 1, j] + v[i - 1, j + 1])
    else:
        Fe = 0.5 * (u[i, j] + u[i + 1, j])
        Fw = 0.5 * (u[i, j - 1] + u[i + 1, j - 1])
        Fn = 0.5 * (v[i, j] + v[i + 1, j])
        Fs = 0.5 * (v[i, j] + v[i - 1, j])
        # print(f"{i} {j}: {Fe} {Fw} {Fn} {Fs}")
    return Fe, Fw, Fn, Fs


def get_coefficients(solver_io: SolverIO, velocity="u") -> None:
    """
    get cofficients after discretizing advection term
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells
    D = solver_io.cfd_geometry.D

    cfd_coefficients = solver_io.cfd_coefficients
    a_e = cfd_coefficients.a_e
    a_w = cfd_coefficients.a_w
    a_n = cfd_coefficients.a_n
    a_s = cfd_coefficients.a_s

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            Fe, Fw, Fn, Fs = get_face_v(u, v, i, j, velocity=velocity)

            a_e[i, j] += max(-Fe, 0.0) * D
            a_w[i, j] += max(Fw, 0.0) * D
            a_s[i, j] += max(Fs, 0.0) * D
            a_n[i, j] += max(-Fn, 0.0) * D
