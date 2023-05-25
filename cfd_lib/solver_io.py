import numpy as np
from typing import Tuple
from dataclasses import dataclass


class CFDGeometry:
    def __init__(self, D: float, num_x_cells: int, num_y_cells: int) -> None:
        self.D = D
        self.num_x_cells = num_x_cells
        self.num_y_cells = num_y_cells
        self.grid_shape = (num_y_cells, num_x_cells)
        self.fluid_grid_shape = (num_y_cells - 2, num_x_cells - 2)


class CFDFields:
    def __init__(
        self, grid_shape: Tuple[int, int], fluid_grid_shape: Tuple[int, int]
    ) -> None:
        self.u = np.zeros(grid_shape)
        self.v = np.zeros(grid_shape)
        self.p = np.zeros(grid_shape)
        self.p_prime = np.zeros(grid_shape)

        self.mass_imb = 100.0 * np.ones(fluid_grid_shape)
        self.u_imb = 100.0 * np.ones(fluid_grid_shape)
        self.v_imb = 100.0 * np.ones(fluid_grid_shape)

    def get_center_u(self) -> np.array:
        """
        get centered u velocity
        """
        u = self.u
        u_center = np.zeros(self.fluid_grid_shape)
        num_y_cell, num_x_cell = self.grid_shape
        i = num_x_cell - 2
        while i != 0:
            u_center[:, i - 1] = 0.5 * (
                u[1 : num_y_cell - 1, i - 1] + u[1 : num_y_cell - 1 :, i]
            )
            i -= 1
        return u_center

    def get_center_v(self) -> np.array:
        """
        get centered v velocity
        """
        v = self.v
        v_center = np.zeros(self.fluid_grid_shape)
        num_y_cell, num_x_cell = self.grid_shape
        j = num_y_cell - 2
        while j != 0:
            v_center[j - 1, :] = 0.5 * (
                v[j - 1, 1 : num_x_cell - 1] + v[j, 1 : num_x_cell - 1]
            )
            j -= 1
        return v_center


class CFDCofficients:
    def __init__(self, grid_shape: Tuple[int, int]) -> None:
        self.ap_u = np.zeros(grid_shape)
        self.ap_v = np.zeros(grid_shape)
        self.ap_p = np.zeros(grid_shape)
        self.a_p = np.zeros(grid_shape)
        self.a_e = np.zeros(grid_shape)
        self.a_w = np.zeros(grid_shape)
        self.a_n = np.zeros(grid_shape)
        self.a_s = np.zeros(grid_shape)
        self.b = np.zeros(grid_shape)

        self.flag_u = -1 * np.ones(grid_shape, dtype=np.int32)
        self.flag_v = -1 * np.ones(grid_shape, dtype=np.int32)
        self.flag_p = -1 * np.ones(grid_shape, dtype=np.int32)

        self.resistance_x = np.zeros(grid_shape)
        self.resistance_y = np.zeros(grid_shape)

    def reinit(self) -> None:
        self.ap_u.fill(0.0)
        self.ap_v.fill(0.0)
        self.ap_p.fill(0.0)

        self.a_p.fill(0.0)
        self.a_e.fill(0.0)
        self.a_w.fill(0.0)
        self.a_n.fill(0.0)
        self.a_s.fill(0.0)
        self.b.fill(0.0)


@dataclass
class SolverControls:
    """Solver Controls"""

    pressure_iterations: int = 300
    velocity_iterations: int = 300
    x_velocity_relaxation: float = 0.7
    y_velocity_relaxation: float = 0.7
    pressure_relaxation: float = 0.3
    max_outer_iterations: int = 80


class SolverIO:
    solver_controls: SolverControls

    def __init__(
        self, D: float, num_x_cells: int, num_y_cells: int, solver_controls=None
    ) -> None:
        self.cfd_geometry = CFDGeometry(D, num_x_cells, num_y_cells)
        self.grid_shape = self.cfd_geometry.grid_shape
        self.fluid_grid_shape = self.cfd_geometry.fluid_grid_shape
        self.cfd_coefficients = CFDCofficients(self.grid_shape)
        self.cfd_fields = CFDFields(self.grid_shape, self.fluid_grid_shape)

        if isinstance(solver_controls, dict):
            self.solver_controls = SolverControls(**solver_controls)
        else:
            self.solver_controls = SolverControls()


if __name__ == "__main__":
    d = 0.1524
    num_x_cells = 16 + 2
    num_y_cells = 4 + 2
    solver_io = SolverIO(d, num_x_cells, num_y_cells)

    print(solver_io.grid_shape)
    print(solver_io.cfd_coefficients.__dict__)
