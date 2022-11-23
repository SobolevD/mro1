import numpy as np

from utils.consts import N


def get_normal_vector(dim, length):
    vector = np.zeros([dim, length], "uint8")

    for step in range(1, length):
        vector = vector + [np.random.uniform(-0.5, 0.5, length), np.random.uniform(-0.5, 0.5, length)]

    return vector / (np.sqrt(length) * np.sqrt(1 / 12))


def get_dataset(vector: object, M: object, cor_matrix: object, size: object) -> object:
    A = np.zeros([2, 2], "float32")
    A[0][0] = np.sqrt(cor_matrix[0][0])
    A[1][0] = cor_matrix[1][0] * np.sqrt(cor_matrix[0][0])
    A[1][1] = np.sqrt(cor_matrix[1][1] - (cor_matrix[0][1] ** 2 / cor_matrix[0][0]))
    A = np.matmul(A, vector)
    X = [A[0] + M[0],
         A[1] + M[1]]
    return np.reshape(X, (2, size))


def get_dataset_l(vector, M, cor_matrix, size):
    if M[0] == 1 and M[1] == -1:
        return np.load('X0_1.npy')
    if M[0] == 2 and M[1] == 2:
        return np.load('X1_1.npy')


def get_dataset_le(vector, M, cor_matrix, size):
    if M[0] == 1 and M[1] == -1:
        return np.load('X0_e.npy')
    if M[0] == 2 and M[1] == 2:
        return np.load('X1_e.npy')
