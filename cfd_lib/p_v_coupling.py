from cfd_lib.solver_io import SolverIO
from cfd_lib.cfd_constants import FLUID, RHO


def SIMPLE(solver_io: SolverIO) -> None:
    """
    get coefficients for SIMPLE
    Semi Implicit Method for Pressure Linked Equations
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
    ap_u = cfd_coefficients.ap_u
    ap_v = cfd_coefficients.ap_v
    a_p = cfd_coefficients.a_p

    flag_u = cfd_coefficients.flag_u
    flag_v = cfd_coefficients.flag_v

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            if flag_u[i, j] == FLUID:
                a_e[i, j] = D * D / ap_u[i, j]
            else:
                a_e[i, j] = 0.0

            if flag_u[i, j - 1] == FLUID:
                a_w[i, j] = D * D / ap_u[i, j - 1]
            else:
                a_w[i, j] = 0.0

            if flag_v[i, j] == FLUID:
                a_n[i, j] = D * D / ap_v[i, j]
            else:
                a_n[i, j] = 0.0

            if flag_v[i - 1, j] == FLUID:
                a_s[i, j] = D * D / ap_v[i - 1, j]
            else:
                a_s[i, j] = 0.0

            # negtive value
            b[i, j] = -1 * RHO * (u[i, j] - u[i, j - 1] + v[i, j] - v[i - 1, j]) * D
            a_p[i, j] = a_e[i, j] + a_w[i, j] + a_n[i, j] + a_s[i, j]


def update_p(solver_io: SolverIO, sor=0.5) -> None:
    """
    update pressures with the pressure corrections
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells
    D = solver_io.cfd_geometry.D

    cfd_fields = solver_io.cfd_fields
    p_prime = cfd_fields.p_prime
    p = cfd_fields.p

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            p[i, j] += sor * p_prime[i, j]


def update_velocity(solver_io: SolverIO):
    """
    update velocities with the pressure corrections
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells
    D = solver_io.cfd_geometry.D

    cfd_coefficients = solver_io.cfd_coefficients
    flag_u = cfd_coefficients.flag_u
    flag_v = cfd_coefficients.flag_v
    ap_u = cfd_coefficients.ap_u
    ap_v = cfd_coefficients.ap_v

    cfd_fields = solver_io.cfd_fields
    p_prime = cfd_fields.p_prime
    u = cfd_fields.u
    v = cfd_fields.v

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            if flag_u[i, j] == FLUID:
                u[i, j] -= D / ap_u[i, j] * (p_prime[i, j + 1] - p_prime[i, j]) / RHO
            if flag_v[i, j] == FLUID:
                v[i, j] -= D / ap_v[i, j] * (p_prime[i + 1, j] - p_prime[i, j]) / RHO
