import matplotlib.pyplot as plt
import numpy as np

from utils.baios import get_bayos_lines
from utils.consts import N, M0, M1, B0, B1
from utils.errors import classification_error
from utils.fisher import get_W, get_wN, get_sigma, get_linear_border
from utils.normal import get_normal_vector, get_dataset_l, get_dataset_le
from utils.robbins_monro import akp, nsko, draw_robbins_monro_line, draw_beta_dependency, \
    draw_W_dependency
from utils.sko_minimize import get_W_sko, get_W__sko

if __name__ == '__main__':

    normal_vector0 = get_normal_vector(2, N)
    normal_vector1 = get_normal_vector(2, N)

    X0 = get_dataset_l(normal_vector0, M0, B0, N)
    X1 = get_dataset_l(normal_vector1, M1, B1, N)

    X0_e = get_dataset_le(normal_vector0, M0, B0, N)
    X1_e = get_dataset_le(normal_vector1, M1, B0, N)

    # 1. Построить линейный классификатор, максимизирующий критерий Фишера, для классов 0 и 1
    # двумерных нормально распределенных векторов признаков для случаев равных и неравных
    # корреляционных матриц. Сравнить качество полученного классификатора с байесовским классификатором

    # =============================
    # Разные корреляционные матрицы
    # =============================
    fisher_W    = get_W(M0, M1, B0, B1)
    sigma_0     = get_sigma(fisher_W, B0)
    sigma_1     = get_sigma(fisher_W, B1)
    fisher_wN   = get_wN(M0, M1, B0, B1, sigma_0, sigma_1)

    fisher_X1   = np.arange(-3, 3, 0.01)
    fisher_X0   = get_linear_border(fisher_W, fisher_X1, fisher_wN)

    x_1_bayes = np.linspace(-3, 3, 400)
    x_00_bayes = np.zeros(len(X0[0]))
    x_01_bayes = np.zeros(len(X0[0]))
    for i in range(0, len(X0[0])):
        x_00_bayes[i], x_01_bayes[i] = get_bayos_lines(B0, B1, M0, M1, 0.5, 0.5, x_1_bayes[i])

    plt.scatter(fisher_X0, fisher_X1)
    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])

    plt.scatter(x_00_bayes, x_1_bayes)
    plt.scatter(x_01_bayes, x_1_bayes)

    plt.xlim((-2, 5))
    plt.title("Критерий Фишера. Разные кор.матрицы")
    plt.show()

    #print(f'Критерий Фишера. Разные кор.матрицы. Ошибка: {classification_error(X0, np.append(fisher_W, fisher_wN), 0)}')

    # =============================
    # Равные корреляционные матрицы
    # =============================
    fisher_W    = get_W(M0, M1, B0, B0)
    sigma_0     = get_sigma(fisher_W, B0)
    sigma_1     = get_sigma(fisher_W, B0)
    fisher_wN   = get_wN(M0, M1, B0, B0, sigma_0, sigma_1)

    fisher_X1_e   = np.arange(-3, 3, 0.01)
    fisher_X0_e   = get_linear_border(fisher_W, fisher_X1_e, fisher_wN)

    x_1_bayes = np.linspace(-3, 3, 400)
    x_0_bayes = np.zeros(len(X0[0]))
    for i in range(0, len(X0[0])):
        x_0_bayes[i] = get_bayos_lines(B0, B0, M0, M1, 0.5, 0.5, x_1_bayes[i])

    plt.scatter(fisher_X0_e, fisher_X1_e)
    plt.scatter(X0_e[0], X0_e[1])
    plt.scatter(X1_e[0], X1_e[1])

    plt.scatter(x_0_bayes, x_1_bayes)

    plt.xlim((-2, 5))
    plt.title("Критерий Фишера. Равные кор.матрицы")
    plt.show()

    print(f'Критерий Фишера. Равные кор.матрицы. Ошибка: {classification_error(X0_e, np.append(fisher_W, fisher_wN), 0)}')

    # 2. Построить линейный классификатор, минимизирующий среднеквадратичную ошибку, для классов 0 и 1
    # двумерных нормально распределенных векторов признаков для случаев равных и неравных корреляционных матриц.
    # Сравнить качество полученного классификатора с байесовским классификатором и классификатором Фишера.

    # =============================
    # Разные корреляционные матрицы
    # =============================
    sko_W = get_W__sko(X0, X1)
    sko_X1 = np.arange(-6, 6, 0.01)
    sko_X0 = get_linear_border([sko_W[0], sko_W[1]], sko_X1, sko_W[2])

    plt.scatter(x_00_bayes, x_1_bayes)
    plt.scatter(x_01_bayes, x_1_bayes)
    plt.plot(sko_X0, sko_X1)
    plt.scatter(X0[0], X0[1])
    plt.scatter(X1[0], X1[1])

    plt.xlim((-2, 5))
    plt.ylim((-5, 5))
    plt.title("Минимизация ско. Разные кор.матрицы")
    plt.show()

    #print(f'Минимизация ско. Разные кор.матрицы. Ошибка: {classification_error_sko(X0_e, sko_W, 0)}')

    # =============================
    # Равные корреляционные матрицы
    # =============================
    sko_W = get_W__sko(X0, X1)
    sko_X1_e = np.arange(-6, 6, 0.01)
    sko_X0_e = get_linear_border([sko_W[0], sko_W[1]], sko_X1, sko_W[2])

    plt.scatter(fisher_X0_e, fisher_X1_e)
    plt.plot(sko_X0, sko_X1)
    plt.scatter(X0_e.T[:, 0], X0_e.T[:, 1])
    plt.scatter(X1_e.T[:, 0], X1_e.T[:, 1])

    plt.scatter(x_0_bayes, x_1_bayes)

    plt.xlim((-2, 5))
    plt.title("Минимизация ско. Равные кор.матрицы")
    plt.show()

    #print(f'Минимизация ско. Равные кор.матрицы. Ошибка: {classification_error(X0_e, sko_W, 0)}')

    # 3. Построить линейный классификатор, основанный на процедуре Роббинса-Монро, для классов 0 и 1 двумерных
    # нормально распределенных векторов признаков для случаев равных и неравных корреляционных матриц. Исследовать
    # зависимость скорости сходимости итерационного процесса и качества классификации от выбора начальных
    # условий и выбора последовательности корректирующих коэффициентов. Сравнить качество полученного
    # классификатора с байесовским классификатором.

    draw_robbins_monro_line(X0, X1, akp, 'Разные кор.матрицы', [x_01_bayes, x_1_bayes])
    draw_robbins_monro_line(X0_e, X1_e, akp, 'Равные кор.матрицы', [x_0_bayes, x_1_bayes])

    draw_robbins_monro_line(X0, X1, nsko, 'Разные кор.матрицы')
    draw_robbins_monro_line(X0_e, X1_e, nsko, 'Равные кор.матрицы')

    draw_beta_dependency(X0, X1, akp, 'Разные кор.матрицы')
    draw_beta_dependency(X0_e, X1_e, akp, 'Равные кор.матрицы')

    draw_beta_dependency(X0, X1, nsko, 'Разные кор.матрицы')
    draw_beta_dependency(X0_e, X1_e, nsko, 'Равные кор.матрицы')

    draw_W_dependency(X0, X1, akp, 'Разные кор.матрицы')
    draw_W_dependency(X0_e, X1_e, akp, 'Равные кор.матрицы')

    draw_W_dependency(X0, X1, nsko, 'Разные кор.матрицы')
    draw_W_dependency(X0_e, X1_e, nsko, 'Равные кор.матрицы')




