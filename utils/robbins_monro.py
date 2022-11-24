import numpy as np
from matplotlib import pyplot as plt

from utils.errors import classification_error


def get_X0(x1, W):
    return -(W[2] + W[1] * x1) / W[0]


def get_linear_border(W, X1):
    func = np.vectorize(get_X0, excluded=['W', 'wN'])
    return func(X1, W=W)


def shuffle(X, R):
    result_len = len(X[0])
    i = np.arange(result_len)
    np.random.shuffle(i)
    return X[:, i], R[:, i]


def alpha_k(k, beta=0.7):
    return 1 / (k ** beta)


def akp(r, W, x):
    WTx = np.matmul(W, x)
    return -x * np.sign((r - WTx))


def nsko(r, W, x):
    WTx = np.matmul(W, x)
    return -x * (r - WTx)


def get_W_RM(X, R, get_alpha, deriv_criterion, N, beta=0.7, w_start=None):

    W = ''

    if w_start is None:
        W = np.random.rand(3)
    else:
        W = np.array((w_start, w_start, w_start))

    for k in range(1, N + 1):
        alpha = get_alpha(k, beta)
        W = W - (alpha * deriv_criterion(R[:, k - 1], W, X[:, k - 1]))

    return W


def __get_X_R(X0, X1):
    np.random.seed(360)
    X = np.concatenate((X0, X1), axis=1)
    ones = np.full((1, len(X[0])), 1)
    X = np.concatenate((X, ones))

    len_R_0 = len(X0[0])
    len_R_1 = len(X1[0])

    ones = np.full((1, len_R_0), 1)
    m_ones = np.full((1, len_R_1), -1)

    R = np.concatenate((m_ones, ones), axis=1)

    X, R = shuffle(X, R)
    return X, R

def draw_robbins_monro_line(X0, X1, derivative_func, title_cor, bayes_coords = None, dif_cor_matrix = True):

    X, R = __get_X_R(X0, X1)

    colors = {500: 'blue', 600: 'blue', 700: 'black', 800: 'red'}
    labels = {500: 'Step 1', 600: 'Step 2', 700: 'Step 3', 800: 'Step 4'}

    W = ''
    for i in range(500, 801, 100):
        W = get_W_RM(X, R, alpha_k, derivative_func, i)
        rm_X1 = np.linspace(-4, 4, 100)

        rm_X0 = get_linear_border(W, rm_X1)
        plt.plot(rm_X0, rm_X1, color=colors[i], label=(labels[i] + '; N = ' + str(i)))

    if derivative_func == nsko:
        plt.title('Robbins monro: NSKO.' + title_cor)
        if dif_cor_matrix:
            print(f'Robbins monro: NSKO. Разные кор.матрицы. Ошибка: {classification_error(X0, W, 0)}')
        else:
            print(f'Robbins monro: NSKO. Равные кор.матрицы. Ошибка: {classification_error(X0, W, 0)}')
    else:
        plt.title('Robbins monro: AKP.' + title_cor)
        if dif_cor_matrix:
            print(f'Robbins monro: AKP. Разные кор.матрицы. Ошибка: {classification_error(X0, W, 0)}')
        else:
            print(f'Robbins monro: AKP. Равные кор.матрицы. Ошибка: {classification_error(X0, W, 0)}')

    plt.legend()
    plt.xlim((-2, 5))
    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])

    if bayes_coords is not None:
        plt.scatter(bayes_coords[0], bayes_coords[1])

    plt.show()


def draw_beta_dependency(X0, X1, derivative_func, title_cor):

    X, R = __get_X_R(X0, X1)

    colors = {1: 'blue', 2: 'blue', 3: 'black', 4: 'red', 5: 'purple'}
    labels = {1: 'Step 1', 2: 'Step 2', 3: 'Step 3', 4: 'Step 4', 5: 'Step 5'}

    beta_intermediate = 0.51
    num = 1
    while beta_intermediate < 1.0:
        W = get_W_RM(X, R, alpha_k, derivative_func, 500, beta_intermediate)
        rm_X1 = np.linspace(-4, 4, 100)

        rm_X0 = get_linear_border(W, rm_X1)
        plt.plot(rm_X0, rm_X1, color=colors[num], label=(labels[num] + '; Beta = ' + '{:.2f}'.format(beta_intermediate)))

        beta_intermediate += 0.1
        num += 1
    if derivative_func == nsko:
        plt.title('Robbins monro: NSKO.' + title_cor)
    else:
        plt.title('Robbins monro: AKP.' + title_cor)

    plt.legend()
    plt.xlim((-2, 5))
    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])
    plt.show()


def draw_W_dependency(X0, X1, derivative_func, title_cor):

    X, R = __get_X_R(X0, X1)

    colors = {1: 'blue', 2: 'blue', 3: 'black', 4: 'red', 5: 'purple'}
    labels = {1: 'Step 1', 2: 'Step 2', 3: 'Step 3', 4: 'Step 4', 5: 'Step 5'}

    W_start = 1
    num = 1
    while W_start < 5:
        W = get_W_RM(X, R, alpha_k, derivative_func, 500, w_start=W_start)
        rm_X1 = np.linspace(-4, 4, 100)

        rm_X0 = get_linear_border(W, rm_X1)
        plt.plot(rm_X0, rm_X1, color=colors[num], label=(labels[num] +
                                                         '; W = (' + '{:.2f}'.format(W_start) +
                                                         ',' + '{:.2f}'.format(W_start) +
                                                         ',' + '{:.2f}'.format(W_start) + ')'))

        W_start += 1
        num += 1
    if derivative_func == nsko:
        plt.title('Robbins monro: NSKO.' + title_cor)
    else:
        plt.title('Robbins monro: AKP.' + title_cor)

    plt.legend()
    plt.xlim((-2, 5))
    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])
    plt.show()