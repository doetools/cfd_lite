import cfd_solver_interface
from cfd_lib.io import save_as_csv

INPUT_PATH = "./reference_projects/"
PROJECT_NAME = "wind_tunnel_with_resistance.json"

solver_io = cfd_solver_interface.run(INPUT_PATH + PROJECT_NAME)

print(solver_io.cfd_fields.mass_imb.max())
print(solver_io.cfd_fields.u_imb.max())
print(solver_io.cfd_fields.v_imb.max())

save_as_csv(solver_io)
