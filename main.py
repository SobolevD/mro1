import numpy as np
import matplotlib.pyplot as plt

from utils.baios import get_line
from utils.normal import get_normal_vector

N = 200
MATRICES_COUNT = 20
GRAPH_SIZE = (10, 10)

COR_MATRIX = np.array(([1, -0.2],
                       [-0.2, 1]))

M1 = [1, -1]
M2 = [2, 2]
M3 = [-1, 1]

def get_sequence_params(vector):
    M = (1 / N) * np.sum(vector, axis=1)

    B = np.zeros([2, 2], "float64")

    for i in range(0, N - 1):
        x_without_m = vector[:, i] - M
        x_without_m = np.reshape(x_without_m, (2, 1))
        B += np.matmul(x_without_m, np.transpose(x_without_m))

    B /= (N - 1)
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
    # normal_vector1 = get_normal_vector(2, N)
    # normal_vector2 = get_normal_vector(2, N)
    #
    # X1 = get_dataset(normal_vector1, M1, COR_MATRIX)
    # X2 = get_dataset(normal_vector2, M2, COR_MATRIX)

    # plt.figure(figsize=GRAPH_SIZE)
    # plt.scatter(X1[0], X1[1])
    # plt.scatter(X2[0], X2[1])
    # plt.show()

    # MC1, BC1 = get_sequence_params(X1)
    # MC2, BC2 = get_sequence_params(X2)
    #
    # result_distance = get_distance(MC1, MC2, COR_MATRIX, COR_MATRIX)
    # print(f'Result distance (task 2): {result_distance}')

    # 3
    b2_1 = np.array((
        [0.5, 0],
        [0, 0.5]))
    b2_2 = np.array((
        [0.4, 0.15],
        [0.15, 0.4]))

    b2_3 = np.array((
        [0.6, -0.2],
        [-0.2, 0.6]))

    X2_1 = get_dataset(get_normal_vector(2, N), M1, b2_1)
    X2_2 = get_dataset(get_normal_vector(2, N), M2, b2_2)
    X2_3 = get_dataset(get_normal_vector(2, N), M3, b2_3)

    # MC2_1, BC2_1 = get_sequence_params(X2_1)
    # MC2_2, BC2_2 = get_sequence_params(X2_2)
    # MC2_3, BC2_3 = get_sequence_params(X2_3)

    # print(f'Result distance between 1-2 (task 3): {get_distance(MC2_1, MC2_2, b2_1, b2_2)}')
    # print(f'Result distance between 1-3 (task 3): {get_distance(MC2_1, MC2_3, b2_1, b2_3)}')
    # print(f'Result distance between 2-3 (task 3): {get_distance(MC2_2, MC2_3, b2_2, b2_3)}')

    x0_12_vector, x1_12_a_vector, x1_12_b_vector = get_line(GRAPH_SIZE, b2_1, b2_2, M1, M2, 1.0/3.0)
    x0_13_vector, x1_13_a_vector, x1_13_b_vector = get_line(GRAPH_SIZE, b2_1, b2_3, M1, M3, 1.0/3.0)
    x0_23_vector, x1_23_a_vector, x1_23_b_vector = get_line(GRAPH_SIZE, b2_2, b2_3, M2, M3, 1.0/3.0)

    plt.figure(figsize=GRAPH_SIZE)
    plt.scatter(X2_1[0], X2_1[1], c='black')
    plt.scatter(X2_2[0], X2_2[1], c='pink')
    plt.scatter(X2_3[0], X2_3[1], c='red')
    plt.scatter(x0_12_vector, x1_12_a_vector, c='green')
    plt.scatter(x0_12_vector, x1_12_b_vector, c='blue')
    plt.scatter(x0_13_vector, x1_13_a_vector, c='orange')
    plt.scatter(x0_13_vector, x1_13_b_vector, c='purple')
    plt.scatter(x0_23_vector, x1_23_a_vector, c='lime')
    plt.scatter(x0_23_vector, x1_23_b_vector, c='yellow')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()