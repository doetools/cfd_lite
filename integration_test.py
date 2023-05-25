import os
import cfd_solver_interface
from cfd_lib.io import (
    read_json,
    check_if_two_json_objects_same,
    get_field_values,
)


def test():
    INPUT_DIR = "./reference_projects/"
    RESULT_DIR = "./expected_results/"

    inputs = os.listdir(INPUT_DIR)
    results = os.listdir(RESULT_DIR)

    for input in inputs:
        if input not in results:
            continue
        solver_io = cfd_solver_interface.run(INPUT_DIR + input)

        fields = get_field_values(solver_io)

        reference = read_json(RESULT_DIR + input)

        assert check_if_two_json_objects_same(
            reference, fields
        ), f"{input.replace('.json','')}, integration test fails"


if __name__ == "__main__":
    test()
