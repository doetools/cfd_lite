from cfd_lib.solver_io import SolverIO
from cfd_lib.cfd_constants import NU, RHO


def get_coefficients(solver_io: SolverIO, velocity="u", fraction_pressure=1.0) -> None:
    """
    get cofficients after discretizing diffusiona and pressure term
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells
    D = solver_io.cfd_geometry.D

    cfd_coefficients = solver_io.cfd_coefficients
    a_e = cfd_coefficients.a_e
    a_w = cfd_coefficients.a_w
    a_n = cfd_coefficients.a_n
    a_s = cfd_coefficients.a_s
    b = cfd_coefficients.b
    resistance_x = cfd_coefficients.resistance_x
    resistance_y = cfd_coefficients.resistance_y
    a_p = cfd_coefficients.a_p

    cfd_fields = solver_io.cfd_fields
    p = cfd_fields.p
    u = cfd_fields.u
    v = cfd_fields.v

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            ## diffusion
            a_e[i, j] = NU
            a_w[i, j] = NU
            a_n[i, j] = NU
            a_s[i, j] = NU
            a_p[i, j] = 0

            ## pressure term

            if velocity == "u":
                b[i, j] = -1 / RHO * (p[i, j + 1] - p[i, j]) * D * fraction_pressure
            else:
                b[i, j] = -1 / RHO * (p[i + 1, j] - p[i, j]) * D * fraction_pressure

            ## source term (lazy-person style linearization)
            # if velocity == "u":
            #     b[i, j] -= 0.5 * resistance_x[i, j] * D * u[i, j] ** 2
            # else:
            #     b[i, j] -= 0.5 * resistance_y[i, j] * D * v[i, j] ** 2

            ## source term (recommended stle linearization)
            if velocity == "u":
                b[i, j] += 0.5 * resistance_x[i, j] * D * u[i, j] ** 2
                a_p[i, j] = resistance_x[i, j] * D * u[i, j]
            else:
                b[i, j] += 0.5 * resistance_y[i, j] * D * v[i, j] ** 2
                a_p[i, j] = resistance_y[i, j] * D * v[i, j]
