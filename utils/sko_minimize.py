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
    return np.matmul(np.transpose(W), z) >= 0


def get_linear_border(W, X1):
    func = np.vectorize(get_X0, excluded=['W'])
    return func(X1, W[2], W=W)
