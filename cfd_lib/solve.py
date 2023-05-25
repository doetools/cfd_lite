from cfd_lib.solver_io import SolverIO
from cfd_lib.cfd_constants import NON_FLUID
import cfd_lib.diffusion_pressure as diffusion_pressure
import cfd_lib.advection_velocity as advection_velocity
from cfd_lib.ap import update_ap_b
from cfd_lib.linear_solvers import gauss_seidel
import cfd_lib.velocity_bc as velocity_bc
import cfd_lib.p_v_coupling as p_v_coupling
import cfd_lib.residuals as residuals


def solve_pipeline(solver_io: SolverIO) -> None:
    """
    main solve loop
    """
    max_outer_iterations = solver_io.solver_controls.max_outer_iterations
    x_velocity_relaxation = solver_io.solver_controls.x_velocity_relaxation
    y_velocity_relaxation = solver_io.solver_controls.y_velocity_relaxation
    pressure_relaxation = solver_io.solver_controls.pressure_relaxation

    go_next_iteration = True
    index = 0

    while max_outer_iterations > index and go_next_iteration:
        ## solve governing equations
        solve_u(solver_io, sor=x_velocity_relaxation)
        solve_v(solver_io, sor=y_velocity_relaxation)
        v_p_couple(solver_io, sor=pressure_relaxation)
        solver_io.cfd_fields.p_prime.fill(0.0)
        index += 1

        ## check every 20 steps if solve stoppable
        if index % 20 == 0:
            go_next_iteration = not is_solve_stoppable(solver_io)

    print(f"has run {index} iterations")


def is_solve_stoppable(
    solver_io: SolverIO, threshold_momentum_imb=0.1, threshold_mass_imb=0.1
) -> bool:
    """
    determine if solve is stoppable by checking the residuals
    """
    residuals.get_mass_imb(solver_io)
    residuals.get_u_imb(solver_io)
    residuals.get_v_imb(solver_io)

    if (
        solver_io.cfd_fields.mass_imb.max() > threshold_mass_imb
        or solver_io.cfd_fields.u_imb.max() > threshold_momentum_imb
        or solver_io.cfd_fields.v_imb.max() > threshold_momentum_imb
    ):
        return False

    return True


def solve_u(solver_io: SolverIO, sor=0.3) -> None:
    """
    solve u velocity
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells

    cfd_coefficients = solver_io.cfd_coefficients
    flag_u = cfd_coefficients.flag_u

    u = solver_io.cfd_fields.u

    diffusion_pressure.get_coefficients(solver_io, velocity="u")
    velocity_bc.set_bc_u(solver_io)
    advection_velocity.get_coefficients(solver_io, velocity="u")
    update_ap_b(solver_io, velocity="u", sor=sor)
    gauss_seidel(
        cfd_coefficients,
        u,
        sor=1.0,
        num_i=num_y_cells - 1,
        num_j=num_x_cells - 2,
        num_iterations=solver_io.solver_controls.velocity_iterations,
        skip_eval=lambda i, j: flag_u[i, j] == NON_FLUID,
    )


def solve_v(solver_io: SolverIO, sor=0.3) -> None:
    """
    solve v velocity
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells

    cfd_coefficients = solver_io.cfd_coefficients
    flag_v = cfd_coefficients.flag_v

    v = solver_io.cfd_fields.v

    diffusion_pressure.get_coefficients(solver_io, velocity="v")
    velocity_bc.set_bc_v(solver_io)
    advection_velocity.get_coefficients(solver_io, velocity="v")
    update_ap_b(solver_io, velocity="v", sor=sor)
    gauss_seidel(
        cfd_coefficients,
        v,
        sor=1.0,
        num_i=num_y_cells - 2,
        num_j=num_x_cells - 1,
        num_iterations=solver_io.solver_controls.velocity_iterations,
        skip_eval=lambda i, j: flag_v[i, j] == NON_FLUID,
    )


def v_p_couple(solver_io: SolverIO, sor=0.5):
    """
    update pressure and velocity after resolving the pressure-velocity link using SIMPLE
    """
    num_y_cells = solver_io.cfd_geometry.num_y_cells
    num_x_cells = solver_io.cfd_geometry.num_x_cells

    cfd_coefficients = solver_io.cfd_coefficients
    flag_p = cfd_coefficients.flag_p

    p_prime = solver_io.cfd_fields.p_prime

    p_v_coupling.SIMPLE(solver_io)
    gauss_seidel(
        cfd_coefficients,
        p_prime,
        sor=1.5,
        num_i=num_y_cells - 1,
        num_j=num_x_cells - 1,
        num_iterations=solver_io.solver_controls.pressure_iterations,
        skip_eval=lambda i, j: flag_p[i, j] == NON_FLUID,
    )

    p_v_coupling.update_p(solver_io, sor=sor)
    p_v_coupling.update_velocity(solver_io)
