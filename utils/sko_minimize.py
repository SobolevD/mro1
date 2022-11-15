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


def get_W__sko(class0, class1):
    size1 = np.shape(class1)
    size0 = np.shape(class0)

    z1Size = ((size1[0] + 1), size1[1])
    z0Size = ((size0[0] + 1), size0[1])

    z1 = np.ones(z1Size)
    z0 = np.ones(z0Size)

    z1[0:size1[0], 0:size1[1]] = class1
    z0[0:size0[0], 0:size0[1]] = class0
    z0 = -1 * z0

    resSize = (3, (size1[1] + size0[1]))
    z = np.ones(resSize)
    z[0:3, 0:z1Size[1]] = z1
    z[0:3, z1Size[1]:resSize[1]] = z0

    tmp = np.linalg.inv(np.matmul(z, np.transpose(z)))

    R = np.ones((resSize[1], 1))
    W = np.matmul(np.matmul(tmp, z), R)

    return np.reshape(W, (3,))


def classification_error_sko(dataset, W, class_id):
    errors = 0
    N = dataset.shape[-1]

    for i in range(N):
        z = np.append(dataset[:, i], 1)
        if classify(z, W) != class_id:
            errors += 1

    return errors / N
