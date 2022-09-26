import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand


N = 200
MATRICES_COUNT = 20
GRAPH_SIZE = (10, 10)

COR_MATRIX = np.array(([6, -0.2],
                       [-0.2, 6]))

M1 = [1, -1]
M2 = [2, 2]
M3 = [-1, 1]


def get_normal_vector(dim, length, accuracy=20):
    vector = np.zeros([dim, length], "uint8")

    for step in range(1, accuracy):
        vector = vector + [np.random.uniform(-0.5, 0.5, length), np.random.uniform(-0.5, 0.5, length)]

    return vector / (np.sqrt(length) * np.sqrt(1 / 12))


def get_sequence_params(vector):
    M = (1 / N) * np.sum(vector, axis=1)

    B = np.zeros([2, 2], "float64")

    for i in range(0, N - 1):
        x_without_m = vector[:, i] - M
        x_without_m = np.reshape(x_without_m, (2, 1))
        B += np.matmul(x_without_m, np.transpose(x_without_m))

    B /= N
    return M, B


def get_dataset(vector, M, cor_matrix):
    A = np.zeros([2, 2], "float32")
    A[0][0] = np.sqrt(cor_matrix[0][0])
    A[1][0] = cor_matrix[1][0] * np.sqrt(cor_matrix[0][0])
    A[1][1] = np.sqrt(cor_matrix[1][1] - (cor_matrix[0][1] ** 2 / cor_matrix[0][0]))
    A = np.matmul(A, vector)
    X = [A[0] + M[0],
         A[1] + M[1]]
    return np.reshape(X, (2, 200))


def get_distance(M0, M1, b0, b1):
    MM = np.subtract(M1, M0)
    MM_transposed = np.transpose(MM)
    if np.array_equal(b0, b1):
        B_neg = np.linalg.inv(b0)
        return np.matmul(np.matmul(MM_transposed, B_neg), MM)
    B_median = (b0 + b1) / 2
    B_median_neg = np.linalg.inv(B_median)
    B_median_det = np.linalg.det(B_median)

    b0_det = np.linalg.det(b0)
    b1_det = np.linalg.det(b1)
    abs_B_mul_sqrt = np.sqrt(b1_det * b0_det)

    result = (1 / 4) * MM_transposed
    result = np.matmul(result, B_median_neg)
    result = np.matmul(result, MM)
    result += np.log(B_median_det / abs_B_mul_sqrt)
    return result


if __name__ == '__main__':

    # 2
    normal_vector1 = get_normal_vector(2, N)
    normal_vector2 = get_normal_vector(2, N)

    X1 = get_dataset(normal_vector1, M1, COR_MATRIX)
    X2 = get_dataset(normal_vector2, M2, COR_MATRIX)

    plt.figure(figsize=GRAPH_SIZE)
    plt.scatter(X1[0], X1[1])
    plt.scatter(X2[0], X2[1])
    plt.show()

    # 3
    b2_1 = np.array((
        [5, 0],
        [0, 5]))
    b2_2 = np.array((
        [6, 0.1],
        [0.1, 6]))
    b2_3 = np.array((
        [7, -0.2],
        [-0.2, 7]))

    X2_1 = get_dataset(get_normal_vector(2, N), M1, b2_1)
    X2_2 = get_dataset(get_normal_vector(2, N), M2, b2_2)
    X2_3 = get_dataset(get_normal_vector(2, N), M3, b2_3)

    plt.figure(figsize=GRAPH_SIZE)
    plt.scatter(X2_1[0], X2_1[1])
    plt.scatter(X2_2[0], X2_2[1])
    plt.scatter(X2_3[0], X2_3[1])
    plt.show()

    MC1, BC1 = get_sequence_params(X1)
    MC2, BC2 = get_sequence_params(X2)

    result_distance = get_distance(MC1, MC2, BC1, BC2)
    print(f'Result distance (task 2): {result_distance}')

    MC2_1, BC2_1 = get_sequence_params(X2_1)
    MC2_2, BC2_2 = get_sequence_params(X2_2)
    MC2_3, BC2_3 = get_sequence_params(X2_3)

    print(f'Result distance between 1-2 (task 3): {get_distance(MC2_1, MC2_2, BC2_1, BC2_2)}')
    print(f'Result distance between 1-3 (task 3): {get_distance(MC2_1, MC2_3, BC2_1, BC2_3)}')
    print(f'Result distance between 2-3 (task 3): {get_distance(MC2_2, MC2_3, BC2_2, BC2_3)}')