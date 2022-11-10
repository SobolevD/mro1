import numpy as np

from utils.fisher import get_X0


def get_z(X):
    z = np.copy(X)
    return np.append(z, 1)


def get_z_neg(X):
    z = np.copy(-X)
    return np.transpose(np.append(z, -1))


def get_W_full(W, wN):
    return np.append(W, wN)


# True - class 0; False - class 1
def classify(z, W):
    return 1 if np.matmul(np.transpose(W), z) >= 0 else 0


def get_linear_border(W, X1):
    func = np.vectorize(get_X0, excluded=['W'])
    return func(X1, W[2], W=W)


def get_sko_training_dataset(X1, X0):
    z = np.concatenate(
        (X1, np.full((1, len(X1)), 1).T), axis=1
    )

    z_neg = np.concatenate(
        (X0 * -1, np.full((1, len(X0)), -1).T), axis=1
    )

    X = np.concatenate((z, z_neg))
    corr_values = np.concatenate(
        (np.full((1, len(z)), 1).T, np.full((1, len(z_neg)), -1).T)
    )

    return X, corr_values


def get_W_sko(X1, X0):
    X, corr_values = get_sko_training_dataset(
        X1, X0
    )
    return (np.linalg.inv(X@ X.T) @ X).T @ corr_values


def classification_error_sko(dataset, W, class_id):
    errors = 0
    N = dataset.shape[-1]

    for i in range(N):
        z = np.append(dataset[:, i], 1)
        if classify(z, W) != class_id:
            errors += 1

    return errors / N
