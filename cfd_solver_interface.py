from cfd_lib.solve import solve_pipeline
from cfd_lib.solver_io import SolverIO
import cfd_lib.residuals as residuals
from cfd_lib.preprocess import process_input
from cfd_lib.io import read_json


def run(input_path="./reference_projects/wind_tunnel_with_hole.json") -> SolverIO:
    # read input
    input = read_json(input_path)

    # process input
    solver_io = process_input(input)

    # solve
    solve_pipeline(solver_io)

    # check imbalance
    residuals.get_mass_imb(solver_io)
    residuals.get_u_imb(solver_io)
    residuals.get_v_imb(solver_io)

    return solver_io


if __name__ == "__main__":
    run()
