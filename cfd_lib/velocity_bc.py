from cfd_lib.solver_io import SolverIO


def set_bc_u(solver_io: SolverIO) -> None:
    """
    assigin bc for solving u
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells

    cfd_coefficients = solver_io.cfd_coefficients
    a_s = cfd_coefficients.a_s
    a_n = cfd_coefficients.a_n

    # symmetric walls
    a_s[1, 1 : num_x_cells - 1] = 0.0
    a_n[num_y_cells - 2, 1 : num_x_cells - 1] = 0.0


def set_bc_v(solver_io: SolverIO) -> None:
    """
    assgining bc for solving v
    """
    pass
