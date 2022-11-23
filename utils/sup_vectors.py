import numpy as np
from qpsolvers import solve_qp


def get_support_vectors(dataset, support_vectors_pos):
    return dataset[support_vectors_pos, :-1, 0]


def get_support_vector_classes(dataset, support_vectors_pos):
    return dataset[support_vectors_pos, -1, 0]


def get_lambda(P, q, G, h, A, b):
    return solve_qp(P, q, G, h, A, b, solver='cvxopt')


def concat(x, value):
    return np.append(x, np.array([[value]]), axis=0)


def get_discr_kernel(support_vectors, lambda_r, x, params):
    sum = 0

    for j in range(support_vectors.shape[0]):
        sum += lambda_r[j] * get_K(
            support_vectors[j].reshape(2, 1),
            x,
            params
        )

    return sum


def get_train_dataset(dataset1, dataset2):

    N = dataset1.shape[2] + dataset2.shape[2]  # 100 + 100 = 200
    dataset_for_train = []

    cycles_count = N // 2

    for i in range(cycles_count):
        dataset_for_train.append(dataset1[:, :, i])

        dataset_for_train[i * 2]     = concat(dataset_for_train[i * 2], -1)  # r (-1)

        dataset_for_train.append(dataset2[:, :, i])

        dataset_for_train[i * 2 + 1] = concat(dataset_for_train[i * 2 + 1], 1)  # r (+1)
    return np.array(dataset_for_train)


def get_linear_classificator(z, W):
    return 1 if get_linear_discriminant(z, W) > 0 else 0


def get_linear_discriminant(z, W):
    return np.matmul(W.T, z)


def get_P(dataset, N):
    P = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = np.matmul(
                dataset[j, :-1, :].T    * get_r(dataset[j, :, :]),
                dataset[i, :-1, :]      * get_r(dataset[i, :, :])
            )
    return P


def get_r(x):
    return x[-1][0]


def get_A(dataset, N):
    A = np.zeros((N,))
    for j in range(N):
        A[j] = get_r(dataset[j])
    return A


def get_K(x, y, params):
    d = params[0]
    c = params[1]
    return pow(np.matmul(x.T, y)[0, 0] + c, d)


def get_P_kernel(dataset, N, params):

    P = np.zeros(shape=(N, N))

    for i in range(N):
        for j in range(N):
            P[i, j] = get_r(dataset[j, :, :]) * \
                      get_r(dataset[i, :, :]) * \
                      get_K(
                          dataset[j, :-1, :],
                          dataset[i, :-1, :],
                          params
                      )
    return P
