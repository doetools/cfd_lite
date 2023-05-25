from cfd_lib.solver_io import SolverIO
from cfd_lib.advection_velocity import get_face_v


def update_ap_b(solver_io: SolverIO, velocity="u", sor=1) -> None:
    """
    aggregate the cofficients for discretized momentum equation and apply the equation-level damping
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

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v

    for i in range(1, num_y_cells - 1):
        for j in range(1, num_x_cells - 1):
            Fe, Fw, Fn, Fs = get_face_v(u, v, i, j, velocity=velocity)
            if velocity == "u":
                phi = u
                save_ap = ap_u
            else:
                phi = v
                save_ap = ap_v

            # get ap
            a_p[i, j] += (
                a_e[i, j] + a_w[i, j] + a_n[i, j] + a_s[i, j] + (Fe - Fw + Fn - Fs) * D
            )

            # damping
            b[i, j] += (1 / sor - 1) * a_p[i, j] * phi[i, j]
            a_p[i, j] /= sor

            # save ap
            save_ap[i, j] = a_p[i, j]
