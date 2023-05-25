from cfd_lib.solver_io import SolverIO
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict


def plot(solver_io: SolverIO) -> None:
    p = solver_io.cfd_fields.p
    x = np.arange(0.25, 8.0, 0.5)
    y = np.arange(0.25, 2.0, 0.5)
    xx, yy = np.meshgrid(x, y, sparse=True)
    h = plt.contourf(x, y, p[1:-1, 1:-1], cmap="rainbow")
    plt.colorbar(orientation="horizontal")

    u_center = solver_io.cfd_fields.get_center_u()
    v_center = solver_io.cfd_fields.get_center_v()
    plt.quiver(xx, yy, u_center, v_center)

    plt.xticks(x)
    plt.yticks(y)
    plt.axis("scaled")
    plt.grid()
    plt.show()


def save_as_csv(solver_io: SolverIO) -> None:
    """
    save the solver_io as csv files
    """
    np.savetxt("cfd_u.csv", solver_io.cfd_fields.u, delimiter=",")
    np.savetxt("cfd_v.csv", solver_io.cfd_fields.v, delimiter=",")
    np.savetxt("cfd_p.csv", solver_io.cfd_fields.p, delimiter=",")


def get_field_values(solver_io: SolverIO) -> Dict:
    """
    get all field values
    """
    fields = {
        "u": solver_io.cfd_fields.u.tolist(),
        "v": solver_io.cfd_fields.v.tolist(),
        "p": solver_io.cfd_fields.p.tolist(),
    }

    return fields


def save_as_json(fields: Dict, save_path="./") -> None:
    """
    save the file as json
    """
    with open(save_path, "w+") as f:
        json.dump(fields, f)


def read_json(input_path: str) -> Dict:
    with open(input_path, "r+") as f:
        content = json.load(f)
    return content


def check_if_two_json_objects_same(json_1: Dict, json_2: Dict) -> bool:
    return json.dumps(json_1, sort_keys=True) == json.dumps(json_2, sort_keys=True)
