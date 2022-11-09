import numpy as np


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