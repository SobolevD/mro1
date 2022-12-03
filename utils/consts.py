import numpy as np

EPS = 1e-04

N = 100
MATRICES_COUNT = 20
GRAPH_SIZE = (10, 10)

M0 = np.array([1, -1])
M1 = np.array([2, 2])
M2 = np.array([-1, 1])
M3 = np.array([1, 1])
M4 = np.array([-1, -1])

B0 = np.array((
    [0.23, -0.18],
    [-0.18, 0.26]))
B1 = np.array((
    [0.22, -0.2],
    [-0.2, 0.22]))
B2 = np.array((
    [0.27, -0.22],
    [-0.22, 0.23]))
B3 = np.array((
    [0.13, -0.1],
    [-0.1, 0.11]))
B4 = np.array((
    [0.13, -0.09],
    [-0.09, 0.13]))