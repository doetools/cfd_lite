from cfd_lib.solver_io import SolverIO
from cfd_lib.cfd_constants import NON_FLUID
from cfd_lib.resistance import get_loss_coefficient
from typing import Dict


def process_input(input: Dict) -> SolverIO:
    """
    process the input and save them in solver_io
    """
    D = input["D"]
    num_x_cells = input["num_x_cells"]
    num_y_cells = input["num_y_cells"]
    solver_controls = input.get("solver_controls")
    solver_io = SolverIO(D, num_x_cells, num_y_cells, solver_controls)

    process_ghost_cells(solver_io)

    inlets = input.get("inlet", [])
    outlets = input.get("outlet", [])

    for fluid_opening in inlets + outlets:
        process_fluid_opening(solver_io, fluid_opening)

    blocks = input.get("block", [])
    for block in blocks:
        process_block(solver_io, block)

    resistance_plates = input.get("resistance_plate", [])
    for resistance_plate in resistance_plates:
        process_resistance_plate(solver_io, resistance_plate)

    return solver_io


def process_resistance_plate(solver_io: SolverIO, plate: Dict) -> None:
    """
    process resistance plate
    """
    cfd_coefficients = solver_io.cfd_coefficients
    resistance_x = cfd_coefficients.resistance_x
    resistance_y = cfd_coefficients.resistance_y

    orientation = plate["orientation"]

    i = plate["x"]
    j = plate["y"]

    if "loss_coefficient" in plate:
        f = plate["loss_coefficient"]
    else:
        beta = plate["open_area"]
        f = get_loss_coefficient(beta)

    if orientation == 0:
        resistance_y[j, i] = f
    else:
        resistance_x[j, i] = f


def process_fluid_opening(solver_io: SolverIO, inlet: Dict) -> None:

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v

    cfd_coefficients = solver_io.cfd_coefficients
    flag_u = cfd_coefficients.flag_u
    flag_v = cfd_coefficients.flag_v

    orientation = inlet["orientation"]
    velocity = inlet["velocity"]
    i = inlet["x"]
    j = inlet["y"]

    if orientation == 0:
        v[j, i] = velocity
        flag_v[j, i] = NON_FLUID
    else:
        u[j, i] = velocity
        flag_u[j, i] = NON_FLUID


def process_block(solver_io: SolverIO, block: Dict) -> None:

    cfd_fields = solver_io.cfd_fields
    u = cfd_fields.u
    v = cfd_fields.v

    cfd_coefficients = solver_io.cfd_coefficients
    flag_u = cfd_coefficients.flag_u
    flag_v = cfd_coefficients.flag_v
    flag_p = cfd_coefficients.flag_p

    i = block["x"]
    j = block["y"]

    flag_p[j, i] = NON_FLUID

    v[j, i] = 0
    flag_v[j, i] = NON_FLUID

    u[j, i] = 0
    flag_u[j, i] = NON_FLUID

    if j - 1 != 0:
        v[j - 1, i] = 0
        flag_v[j - 1, i] = NON_FLUID

    if i - 1 != 0:
        u[j, i - 1] = 0
        flag_u[j, i - 1] = NON_FLUID


def process_ghost_cells(solver_io: SolverIO) -> None:
    cfd_coefficients = solver_io.cfd_coefficients
    flag_u = cfd_coefficients.flag_u
    flag_v = cfd_coefficients.flag_v
    flag_p = cfd_coefficients.flag_p

    cfd_geometry = solver_io.cfd_geometry
    num_x_cells = cfd_geometry.num_x_cells
    num_y_cells = cfd_geometry.num_y_cells

    for flag in [flag_v, flag_u, flag_p]:
        flag[:, 0] = NON_FLUID
        flag[:, num_x_cells - 1] = NON_FLUID
        flag[0, :] = NON_FLUID
        flag[num_y_cells - 1, :] = NON_FLUID

    flag_u[:, num_x_cells - 2] = NON_FLUID
    flag_v[num_y_cells - 2, :] = NON_FLUID
