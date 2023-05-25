import numpy as np
from munch import DefaultMunch
import scipy.sparse as sparse
from scipy import io
from scipy.sparse.linalg import spsolve

from typing import Tuple
import os

FLUID = -1
MULTI_FACE = 1


def read_para_from_file(
    filename="para.txt",
    dir="/home/sesa461392/Documents/Projects/2022_jan/ecostream/test_cfd_coefficients/",
):
    with open(dir + filename, "r") as f:
        c = f.readlines()

    imax, jmax, kmax = c[1].split()
    return int(imax), int(jmax), int(kmax)


def read_coeff_from_file(
    filename="coefficients.txt",
    dir="/home/sesa461392/Documents/Projects/2022_jan/ecostream/test_cfd_coefficients/",
):
    with open(dir + filename, "r") as f:
        c = f.readlines()

    # remove first line
    names = c[0].split()

    # data
    c.pop(0)
    data = np.loadtxt(c)

    coeff = {}

    for i in range(len(names)):
        coeff.update({names[i]: data[:, i]})

    return coeff


class GRID:
    def __init__(self, imax, jmax, kmax, flag):
        self.imax = imax
        self.jmax = jmax
        self.kmax = kmax
        self.flag = flag
        self.size = (imax + 2) * (jmax + 2) * (kmax + 2)
        self._private_full = [0 for i in range(self.size)]
        self.mapping = self.obtain_mapping()

        (self.sparse_size,) = self.mapping.shape

    def IX(self, i, j, k) -> int:
        IMAX = self.imax + 2
        IJMAX = (self.imax + 2) * (self.jmax + 2)
        return (i) + (IMAX) * (j) + (IJMAX) * (k)

    def ijk(self, index) -> int:
        IMAX = self.imax + 2
        IJMAX = (self.imax + 2) * (self.jmax + 2)

        k = int(index / IJMAX)
        res = index % IJMAX
        j = int(res / IMAX)
        i = res % IMAX

        return i, j, k

    def obtain_mapping(self) -> np.array:
        map = []
        count = 0
        for k in range(self.kmax + 2):
            for j in range(self.jmax + 2):
                for i in range(self.imax + 2):
                    ix = self.IX(i, j, k)
                    if int(self.flag[ix]) == FLUID:
                        map.append(ix)
                        self._private_full[ix] = count
                        count += 1

        return np.asarray(map)

    def dense_to_sparse_index(self, i):
        # loc = np.where(self.mapping == i)[0]

        # if len(loc) != 1:
        #     raise Warning(f"index {i} is not found or found multiple in mapping array")

        # return loc[0]

        """
        use space to speed up search
        """
        return self._private_full[i]

    def sparse_to_dense_matrix(self, i):
        return self.mapping[i]

    def extract_sparse_matrix(self, x):
        return np.take(x, self.mapping.tolist())


def obtain_ab_csr(coeff, grid: GRID) -> Tuple[sparse.csr_matrix, np.array]:
    def get_west_x(coeff, ind) -> float:
        ttype = coeff.ttype
        x = coeff.x
        te = coeff.te

        if ttype[ind] == MULTI_FACE:
            return te[ind]
        return x[ind]

    def get_east_x(coeff, ind) -> float:
        ttype = coeff.ttype
        x = coeff.x
        tw = coeff.tw

        if ttype[ind] == MULTI_FACE:
            return tw[ind]
        return x[ind]

    def get_north_x(coeff, ind) -> float:
        ttype = coeff.ttype
        x = coeff.x
        ts = coeff.ts

        if ttype[ind] == MULTI_FACE:
            return ts[ind]
        return x[ind]

    def get_south_x(coeff, ind) -> float:
        ttype = coeff.ttype
        x = coeff.x
        tn = coeff.tn

        if ttype[ind] == MULTI_FACE:
            return tn[ind]
        return x[ind]

    def get_front_x(coeff, ind) -> float:
        ttype = coeff.ttype
        x = coeff.x
        tb = coeff.tb
        if ttype[ind] == MULTI_FACE:
            return tb[ind]
        return x[ind]

    def get_back_x(coeff, ind) -> float:
        ttype = coeff.ttype
        x = coeff.x
        tf = coeff.tf

        if ttype[ind] == MULTI_FACE:
            return tf[ind]
        return x[ind]

    data = []
    indprt = [0]
    indices = []

    sp_b = np.zeros(grid.sparse_size, dtype=np.float32)

    IX = grid.IX
    spind = grid.dense_to_sparse_index
    flag = coeff.flag
    ap = coeff.ap
    b = coeff.b
    ae = coeff.ae
    aw = coeff.aw
    an = coeff.an
    a_s = coeff.a_s
    af = coeff.af
    ab = coeff.ab

    for index in grid.mapping:
        i, j, k = grid.ijk(index)
        it_id = IX(i, j, k)
        nnz = 0
        b_tmp = b[it_id]

        # check
        if flag[it_id] != FLUID:
            raise Warning(f"cell[{i},{j},{k}] has wrong type")

        # back
        back_id = IX(i, j, k - 1)
        if flag[back_id] == FLUID:
            # flip the direction
            psi = -1 * ab[it_id]
            if psi != 0:
                data.append(psi)
                indices.append(spind(back_id))
                nnz += 1
        else:
            x = get_back_x(coeff, back_id)
            b_tmp += ab[it_id] * x
        # south
        south_id = IX(i, j - 1, k)
        if flag[south_id] == FLUID:
            # flip the direction
            psi = -1 * a_s[it_id]
            if psi != 0:
                data.append(psi)
                indices.append(spind(south_id))
                nnz += 1
        else:
            x = get_south_x(coeff, south_id)
            b_tmp += a_s[it_id] * x

        # west
        west_id = IX(i - 1, j, k)
        if flag[west_id] == FLUID:
            # flip the direction
            psi = -1 * aw[it_id]
            if psi != 0:
                data.append(psi)
                indices.append(spind(west_id))
                nnz += 1
        else:
            x = get_west_x(coeff, west_id)
            b_tmp += aw[it_id] * x

        # itself
        data.append(ap[it_id])
        indices.append(spind(it_id))
        nnz += 1

        # east
        east_id = IX(i + 1, j, k)
        if flag[east_id] == FLUID:
            # flip the direction
            psi = -1 * ae[it_id]
            if psi != 0:
                data.append(psi)
                indices.append(spind(east_id))
                nnz += 1
        else:
            x = get_east_x(coeff, east_id)
            b_tmp += ae[it_id] * x

        # north
        north_id = IX(i, j + 1, k)
        if flag[north_id] == FLUID:
            # flip the direction
            psi = -1 * an[it_id]
            if psi != 0:
                data.append(psi)
                indices.append(spind(north_id))
                nnz += 1
        else:
            x = get_north_x(coeff, north_id)
            b_tmp += an[it_id] * x

        # front
        front_id = IX(i, j, k + 1)
        if flag[front_id] == FLUID:
            # flip the direction
            psi = -1 * af[it_id]
            if psi != 0:
                data.append(psi)
                indices.append(spind(front_id))
                nnz += 1
        else:
            x = get_front_x(coeff, front_id)
            b_tmp += af[it_id] * x

        ## update indprt
        last = indprt[-1]
        new = last + nnz
        indprt.append(new)

        ## update sp_b
        sp_b[spind(it_id)] = b_tmp

        # if spind(index) == 1:
        #     print(data, indices, indprt)

    data, indprt, indices = (
        np.asarray(data, dtype=np.float32),
        np.asarray(indprt, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
    )

    sp_a = sparse.csr_matrix(
        (data, indices, indprt), shape=(grid.sparse_size, grid.sparse_size)
    )

    return sp_a, sp_b


if __name__ == "__main__":

    imax, jmax, kmax = read_para_from_file()
    coeff = DefaultMunch.fromDict(read_coeff_from_file())
    grid = GRID(imax, jmax, kmax, coeff.flag)
    a, b = obtain_ab_csr(coeff, grid)
    print("get matrix a and vector b")

    # convert to csc from csr
    a = a.tocsc()
    print("convert matrix a into csc")

    # write to files
    io.mmwrite(
        "example_a.mtx",
        a,
        comment="",
        field="real",
    )

    # print(b.max(), b.min())
    np.savetxt("example_b.txt", b, fmt="%10.8f")

    print("write a and b into files")

    ## test Ax = 1
    # ones = np.ones(grid.sparse_size, dtype=np.float32)
    # x_test = spsolve(a, ones)
    # print(x_test)

    ## test Ax = b
    x_ref = grid.extract_sparse_matrix(coeff.x)
    x = spsolve(a, b)
    print(np.allclose(x_ref, x, rtol=1e-3, atol=1e-3))
    res = np.abs(x - x_ref)
    print(res.max(), res.min(), res.mean())

    ## test Ax = b using GLU
    os.system(
        "./sparse_matrix/GLU_public-master/src/lu_cmd -i ./example_a.mtx ./example_b.txt"
    )
    x_glu = np.loadtxt("./x.dat")
    print(x_glu)
    print(x)
    print(x_ref)

    print(np.allclose(x_glu, x))


# print(x_ref.max(), x_ref.min())
# print(grid.mapping)
# print(grid.dense_to_sparse_index(1137))
# print(coeff.ae.shape, grid.size)
