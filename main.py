import matplotlib.pyplot as plt
import numpy as np

from utils.consts import N, M0, M1, B0, B1
from utils.fisher import get_W, get_wN, get_sigma, get_linear_border
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
