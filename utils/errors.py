import numpy as np

from utils.baios import __mul_3_matrices


def classification_error(dataset, W, class_id):
    errors = 0  # показывает число неверно определенных элементов
    N = dataset.shape[-1]

    for i in range(N):
        z = np.append(dataset[:, i], 1)
        if linear_classificator(z, W) != class_id:
            errors += 1

    return errors / N


def linear_classificator(z, W):
    return 1 if linear_discriminant(z, W) > 0 else 0


def linear_discriminant(z, W):
    return np.matmul(W, z)


def classification_error_by_params(X1, M1, M2, B1, B2, P1, P2):
    M1 = M1.reshape(2, 1)
    M2 = M2.reshape(2, 1)

    B1_det = np.linalg.det(B1)
    B1_inv = np.linalg.inv(B1)

    B2_det = np.linalg.det(B2)
    B2_inv = np.linalg.inv(B2)

    p1_log = np.log(P1)
    p2_log = np.log(P2)

    elements_count = len(X1[0])

    errors_count = 0
    for i in range(0, elements_count):
        x = X1[:, i].reshape(2, 1)

        d1_part1 = - np.log(np.sqrt(B1_det))
        d1_part2 = - 0.5 * __mul_3_matrices((x - M1).T, B1_inv, (x - M1))

        d1 = np.float(p1_log + d1_part1 + d1_part2)

        d2_part1 = - np.log(np.sqrt(B2_det))
        d2_part2 = - 0.5 * __mul_3_matrices((x - M2).T, B2_inv, (x - M2))
        d2 = np.float_(p2_log + d2_part1 + d2_part2)

        if d2 > d1:
            errors_count += 1

    return errors_count / elements_count
