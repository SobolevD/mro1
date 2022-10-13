import numpy as np
import matplotlib.pyplot as plt

from utils.baios import get_line, baios_classifier
import utils
from utils.laplas import inv_laplas_function, calculate_Mahalanobis_distance, get_p_error
from utils.minimax import get_mn_line
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




def Neyman_Pearson(M0, M1, b, p, k):
    b01 = b[0][1]
    b00 = b[0][0]
    b10 = b[1][0]

    m0 = M0[0]
    m1 = M0[1]

    _m0 = M1[0]
    _m1 = M1[1]

    laplas = inv_laplas_function(1 - p)
    rho = calculate_Mahalanobis_distance(M0, M1, b)

    x = (-k * (b01 * (-m0 + _m0) + b10 * (-m1 + _m1) - 1 / 2 * rho - np.sqrt(rho) * laplas)) / (
            b00 * (-m0 + _m0) + b10 * (-m1 + _m1))
    return x


def get_Neyman_Pearson_line(shape, b, m1, m2, p):
    step = 0.005
    points_count = shape[0] / step

    x0_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))
    x1_a_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))

    counter = 0
    i = -(shape[0] / 2.0)
    while i < shape[1] / 2.0 - step:
        x0_vector[counter] = i
        x1_vector = Neyman_Pearson(m1, m2, b, p, i)
        x1_a_vector[counter] = x1_vector[0]
        counter += 1
        i += step
    return x0_vector, x1_a_vector


if __name__ == '__main__':
    # 2
    normal_vector1 = get_normal_vector(2, N)
    normal_vector2 = get_normal_vector(2, N)
    #
    X1 = get_dataset(normal_vector1, M1, COR_MATRIX)
    X2 = get_dataset(normal_vector2, M2, COR_MATRIX)

    plt.figure(figsize=GRAPH_SIZE)
    plt.title("Baios for 2")
    plt.scatter(X1[0], X1[1])
    plt.scatter(X2[0], X2[1])
    y_bs_vector, x_bs_vector = get_line(GRAPH_SIZE, COR_MATRIX, COR_MATRIX, M1, M2, 0.5)
    plt.scatter(x_bs_vector, y_bs_vector, c='green')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()

    plt.figure(figsize=GRAPH_SIZE)
    plt.title("Neuman Pearson")
    plt.scatter(X1[0], X1[1])
    plt.scatter(X2[0], X2[1])

    y_np_vector, x_np_vector = get_Neyman_Pearson_line(GRAPH_SIZE, COR_MATRIX, M1, M2, 0.05)
    plt.scatter(x_np_vector, y_np_vector, c='green')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()

    C = np.array([[0, 1], [1, 0]])
    print(get_p_error(M1, M2, COR_MATRIX, C, 0.5, 0.5))
    print(np.sum(get_p_error(M1, M2, COR_MATRIX, C, 0.5, 0.5)) / 2)

    plt.figure(figsize=GRAPH_SIZE)
    plt.title("Minimax")
    plt.scatter(X1[0], X1[1])
    plt.scatter(X2[0], X2[1])
    y_mn_vector, x_mn_vector = get_mn_line(GRAPH_SIZE, COR_MATRIX, C, M1, M2)
    plt.scatter(x_mn_vector, y_mn_vector, c='green')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()

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

    x0_12_vector, x1_12_a_vector, x1_12_b_vector = get_line(GRAPH_SIZE, b2_1, b2_2, M1, M2, 1.0 / 3.0)
    x0_13_vector, x1_13_a_vector, x1_13_b_vector = get_line(GRAPH_SIZE, b2_1, b2_3, M1, M3, 1.0 / 3.0)
    x0_23_vector, x1_23_a_vector, x1_23_b_vector = get_line(GRAPH_SIZE, b2_2, b2_3, M2, M3, 1.0 / 3.0)

    plt.figure(figsize=GRAPH_SIZE)
    plt.title("Baios for 3")
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

    sum = 0
    for i in range(0, 99):
        sum += baios_classifier(0.5, 0.5, b2_1, b2_2, M1, M2, np.array([X1[0][i], X1[1][i]]))
    print(sum / 100)

    p_max = 0.05
    p = 1.0
    i = 0
    sum = 0
    while (p >= p_max):
        sum += baios_classifier(0.5, 0.5, b2_1, b2_2, M1, M2, np.array([X1[0][i], X1[1][i]]))
        i += 1
        p = sum / i
    print(f'Объем обучающей выборки: {i}')
