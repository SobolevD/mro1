import numpy as np

from utils.baios import __mul_3_matrices


def get_W(M0, M1, B0, B1):
    inverse_component = np.linalg.inv(0.5 * (B0 + B1))
    return np.matmul(inverse_component, (M1 - M0))


def get_wN(M0, M1, B0, B1, sigma_0, sigma_1):
    M1_M0_transpose = np.transpose(M1 - M0)
    half_B1_B0_inverse = np.linalg.inv(0.5 * (B0 + B1))
    sig1_M0_sig0_M1 = (sigma_1 ** 2) * M0 + (sigma_0 ** 2) * M1

    nominator = __mul_3_matrices(M1_M0_transpose, half_B1_B0_inverse, sig1_M0_sig0_M1)
    denominator = sigma_1 ** 2 + sigma_0 ** 2
    return -(nominator / denominator)


# True - class 0, False - class 1
def classify(W, wN, X):
    return np.sum(W * X) + wN > 0


def get_sigma(W, B):
    W_transpose = np.transpose(W)
    return __mul_3_matrices(W_transpose, B, W)


def get_X0(x1, wN, W):
    return -(wN + W[1] * x1) / W[0]


def get_linear_border(W, X1, wN):
    func = np.vectorize(get_X0, excluded=['W', 'wN'])
    return func(X1, wN=wN, W=W)
