import math

import numpy as np

from utils.baios import __mul_3_matrices


def get_cov_B(dataset):
    return np.cov(dataset)


def get_Pa(datasets):
    N_values = []
    for dataset in datasets:
        N_values.append(dataset.shape[1])

    P_values = []
    for dataset in datasets:
        P_values.append(dataset.shape[1] / np.sum(N_values))

    return np.array(P_values)


def get_h(dataset, k):
    n = dataset.shape[0]
    N = dataset.shape[1]
    return pow(N, -k / n)


def get_f_estimate(x, train_dataset):

    n = train_dataset.shape[0]
    N = train_dataset.shape[1]
    k = 0.25
    B = get_cov_B(train_dataset)
    h = get_h(train_dataset, k)
    h = -0.5 * pow(h, -2)

    sqrt_B_det = np.sqrt(np.linalg.det(B))

    const = 1 / (pow(2 * math.pi, n / 2.0) * sqrt_B_det * pow(h, n))

    B_inv = np.linalg.inv(B)
    sum = 0
    for i in range(0, N):
        xi = train_dataset[:, i]
        sum += const * np.exp(h * __mul_3_matrices((x - xi).reshape(1, 2), B_inv, (x - xi).reshape(1, 2).T))
    return sum / N


def parzen_classifier(x, datasets, args=0):
    P_values = get_Pa(datasets)

    f_array = []
    for dataset in datasets:
        f_array.append(get_f_estimate(x, dataset))
    f_array = np.array(f_array)[:, 0]

    result = P_values * f_array
    result = result[:, 0]

    return 0 if result[0] > result[1] else 1
