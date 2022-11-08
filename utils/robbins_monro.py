import numpy as np


def shuffle(X, R):
    result_len = len(X[0])
    i = np.arange(result_len)
    np.random.shuffle(i)
    return X[:, i], R[:, i]


def alpha_k(k, beta=0.7):
    return 1 / (k ** beta)


def akp(r, W, x):
    W_t = np.transpose(W)
    WTx = np.matmul(W_t, x)
    return x * np.sign((r - WTx))


def nsko(r, W, x):
    W_t = np.transpose(W)
    WTx = np.matmul(W_t, x)
    return -(x * (r - WTx))


def get_W_RM(X, R, get_alpha, deriv_criterion, N):
    W = np.random.rand(3)

    for k in range(1, N + 1):
        alpha = get_alpha(k)
        W = W - (alpha * deriv_criterion(R[:, k - 1], W, X[:, k - 1]))

    return W