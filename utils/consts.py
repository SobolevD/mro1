import numpy as np

N = 100
MATRICES_COUNT = 20
GRAPH_SIZE = (10, 10)

COR_MATRIX = np.array(([1, -0.2],
                       [-0.2, 1]))

C = np.array([[0, 1], [1, 0]])

M0 = np.array([1, -1])
M1 = np.array([2, 2])

B0 = np.array((
    [0.43, -0.2],
    [-0.2, 0.56]))
B1 = np.array((
    [0.4, 0.15],
    [0.15, 0.56]))