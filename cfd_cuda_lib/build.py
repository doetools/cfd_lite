from pytools import memoize
from pycuda.compiler import SourceModule


def build_kenerls(file_name="./cfd_cuda_lib/utility.cu"):

    with open(file_name, "r") as f:
        src = f.read()

    kernels = compile(src)

    return kernels


@memoize
def compile(src: str):
    kernels = SourceModule(src)
    return kernels
