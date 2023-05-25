from cfd_lib.solver_io import SolverIO
from cfd_lib.cfd_constants import NU, RHO
from cfd_lib.cfd_constants import NON_FLUID
from cfd_lib.advection_velocity import get_face_v


def get_mass_imb(solver_io: SolverIO) -> None:
    """
    check mass imbalance
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells
    D = solver_io.cfd_geometry.D

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v
    mass_imb = cfd_fields.mass_imb

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            imb = (u[i, j] - u[i, j - 1] + v[i, j] - v[i - 1, j]) * D
            mass_imb[i - 1, j - 1] = abs(imb)


def get_u_imb(solver_io: SolverIO) -> None:
    """
    check imbalance of solving momentum u
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells
    D = solver_io.cfd_geometry.D

    cfd_coefficients = solver_io.cfd_coefficients
    flag_u = cfd_coefficients.flag_u
    resistance_x = cfd_coefficients.resistance_x

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v
    p = cfd_fields.p
    u_imb = cfd_fields.u_imb

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            if flag_u[i, j] == NON_FLUID:
                u_imb[i - 1, j - 1] = 0.0
                continue

            # diff
            imb_diff = (
                NU
                * (
                    (u[i, j + 1] - u[i, j])
                    - (u[i, j] - u[i, j - 1])
                    + (u[i + 1, j] - u[i, j])
                    - (u[i, j] - u[i - 1, j])
                )
                / (D**2)
            )

            # pressure + source
            imb_p = (
                -1 / RHO * (p[i, j + 1] - p[i, j]) / D
                - 0.5 * resistance_x[i, j] * u[i, j] ** 2 / D
            )

            # adv
            Fe, Fw, Fn, Fs = get_face_v(u, v, i, j, velocity="u")
            Ue = u[i, j] if Fe > 0 else u[i, j + 1]
            Uw = u[i, j - 1] if Fw > 0 else u[i, j]
            Un = u[i, j] if Fn > 0 else u[i + 1, j]
            Us = u[i - 1, j] if Fs > 0 else u[i, j]

            imb_adv = (Fe * Ue - Fw * Uw + Fn * Un - Fs * Us) / D

            # save
            u_imb[i - 1, j - 1] = abs(imb_adv - imb_diff - imb_p)


def get_v_imb(solver_io: SolverIO) -> None:
    """
    check imbalance of solving momentum v
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells
    D = solver_io.cfd_geometry.D

    cfd_coefficients = solver_io.cfd_coefficients
    flag_v = cfd_coefficients.flag_v
    resistance_y = cfd_coefficients.resistance_y

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v
    p = cfd_fields.p
    v_imb = cfd_fields.v_imb

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            if flag_v[i, j] == NON_FLUID:
                v_imb[i - 1, j - 1] = 0.0
                continue

            # diff
            imb_diff = (
                NU
                * (
                    (v[i, j + 1] - v[i, j])
                    - (v[i, j] - v[i, j - 1])
                    + (v[i + 1, j] - v[i, j])
                    - (v[i, j] - v[i - 1, j])
                )
                / (D**2)
            )

            # pressure
            imb_p = (
                -1 / RHO * (p[i + 1, j] - p[i, j]) / D
                - 0.5 * resistance_y[i, j] * v[i, j] ** 2 / D
            )

            # adv
            Fe, Fw, Fn, Fs = get_face_v(u, v, i, j, velocity="v")
            Ue = v[i, j] if Fe > 0 else v[i, j + 1]
            Uw = v[i, j - 1] if Fw > 0 else v[i, j]
            Un = v[i, j] if Fn > 0 else v[i + 1, j]
            Us = v[i - 1, j] if Fs > 0 else v[i, j]

            imb_adv = (Fe * Ue - Fw * Uw + Fn * Un - Fs * Us) / D

            # save
            v_imb[i - 1, j - 1] = abs(imb_adv - imb_diff - imb_p)
