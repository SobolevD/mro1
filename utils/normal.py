import numpy as np

from utils.consts import N


def get_normal_vector(dim, length):
    vector = np.zeros([dim, length], "uint8")

    for step in range(1, length):
        vector = vector + [np.random.uniform(-0.5, 0.5, length), np.random.uniform(-0.5, 0.5, length)]

    return vector / (np.sqrt(length) * np.sqrt(1 / 12))


def get_dataset(vector, M, cor_matrix, size):
    A = np.zeros([2, 2], "float32")
    A[0][0] = np.sqrt(cor_matrix[0][0])
    A[1][0] = cor_matrix[1][0] * np.sqrt(cor_matrix[0][0])
    A[1][1] = np.sqrt(cor_matrix[1][1] - (cor_matrix[0][1] ** 2 / cor_matrix[0][0]))
    A = np.matmul(A, vector)
    X = [A[0] + M[0],
         A[1] + M[1]]
    return np.reshape(X, (2, size))
