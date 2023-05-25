import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])

# i th column
print(a[:, 0])


# read arrays
b = np.loadtxt("a_tmp.txt")


print(b[:, 0])
