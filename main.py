import matplotlib.pyplot as plt
import numpy as np

from utils.consts import N, M0, M1, B0, B1
from utils.fisher import get_W, get_wN, get_sigma, get_linear_border
from utils.robbins_monro import get_W_RM, alpha_k, nsko, akp, shuffle
from utils.sko_minimize import get_linear_border as glb
from utils.normal import get_normal_vector, get_dataset
from utils.sko_minimize import get_z, get_z_neg, get_W_full

if __name__ == '__main__':
    normal_vector0 = get_normal_vector(2, N)
    normal_vector1 = get_normal_vector(2, N)

    X0 = get_dataset(normal_vector0, M0, B0, N)
    X1 = get_dataset(normal_vector1, M1, B1, N)

    # 1. Построить линейный классификатор, максимизирующий критерий Фишера, для классов 0 и 1
    # двумерных нормально распределенных векторов признаков для случаев равных и неравных
    # корреляционных матриц. Сравнить качество полученного классификатора с байесовским классификатором
    fisher_W    = get_W(M0, M1, B0, B1)
    sigma_0     = get_sigma(fisher_W, B0)
    sigma_1     = get_sigma(fisher_W, B1)
    fisher_wN   = get_wN(M0, M1, B0, B1, sigma_0, sigma_1)

    fisher_X1   = np.arange(-3, 3, 0.01)
    fisher_X0   = get_linear_border(fisher_W, fisher_X1, fisher_wN)

    plt.scatter(fisher_X0, fisher_X1)
    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])
    plt.xlim((-2, 5))
    plt.show()

    # 2. Построить линейный классификатор, минимизирующий среднеквадратичную ошибку, для классов 0 и 1
    # двумерных нормально распределенных векторов признаков для случаев равных и неравных корреляционных матриц.
    # Сравнить качество полученного классификатора с байесовским классификатором и классификатором Фишера.
    z       = get_z(X0)
    z_neg   = get_z_neg(X0)
    sko_W   = get_W_full(fisher_W, fisher_wN)
    SKO_X0  = glb(sko_W, fisher_X1)

    plt.scatter(SKO_X0, fisher_X1)
    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])
    plt.xlim((-2, 5))
    plt.show()

    # 3. Построить линейный классификатор, основанный на процедуре Роббинса-Монро, для классов 0 и 1 двумерных
    # нормально распределенных векторов признаков для случаев равных и неравных корреляционных матриц. Исследовать
    # зависимость скорости сходимости итерационного процесса и качества классификации от выбора начальных
    # условий и выбора последовательности корректирующих коэффициентов. Сравнить качество полученного
    # классификатора с байесовским классификатором.
    X = np.concatenate((X0, X1), axis=1)
    ones = np.full((1, len(X[0])), 1)
    X = np.concatenate((X, ones))

    len_R_0 = len(X0[0])
    len_R_1 = len(X1[0])
    ones = np.full((1, len_R_0), 1)
    m_ones = np.full((1, len_R_1), -1)
    R = np.concatenate((ones, m_ones), axis=1)

    X, R = shuffle(X, R)

    for i in range(400, 800, 100):
        W = get_W_RM(X, R, alpha_k, nsko, i)
        rm_X1 = np.arange(-3, 3, 0.01)

        rm_X0 = get_linear_border((W[0], W[1]), rm_X1, W[2])

        plt.scatter(rm_X0, rm_X1)

        plt.xlim((-2, 5))

    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])
    plt.show()