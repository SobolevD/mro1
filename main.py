import numpy as np
import matplotlib.pyplot as plt

from utils.baios import get_baios_line, baios_classifier, get_experimental_probability
from utils.canvas import draw_for_two_vectors
from utils.consts import COR_MATRIX, N, M1, M2, GRAPH_SIZE, b2_2, b2_1, b2_3, M3, C
from utils.laplas import get_p_error
from utils.minimax import get_mn_line
from utils.normal import get_normal_vector, get_dataset
from utils.pearson import get_neuman_pearson_line

if __name__ == '__main__':
    # 2
    normal_vector1 = get_normal_vector(2, N)
    normal_vector2 = get_normal_vector(2, N)
    #
    X1 = get_dataset(normal_vector1, M1, COR_MATRIX, N)
    X2 = get_dataset(normal_vector2, M2, COR_MATRIX, N)

    y_bs_vector, x_bs_vector = get_baios_line(GRAPH_SIZE, COR_MATRIX, COR_MATRIX, M1, M2, 0.5)
    y_np_vector, x_np_vector = get_neuman_pearson_line(GRAPH_SIZE, COR_MATRIX, M1, M2, 0.05)
    y_mn_vector, x_mn_vector = get_mn_line(GRAPH_SIZE, COR_MATRIX, C, M1, M2)

    p_err = get_p_error(M1, M2, COR_MATRIX, C, 0.5, 0.5)
    print(f'(Task 1) Вероятность ошибочной классификации для : {p_err}')
    print(f'(Task 1) Суммарная вероятность ошибочной классификации: {np.sum(p_err) / 2.0}')

    draw_for_two_vectors("Baios for 2", GRAPH_SIZE, X1, X2, x_bs_vector, y_bs_vector)
    draw_for_two_vectors("Neuman Pearson", GRAPH_SIZE, X1, X2, x_np_vector, y_np_vector)
    draw_for_two_vectors("Minimax", GRAPH_SIZE, X1, X2, x_mn_vector, y_mn_vector)

    # 3

    X2_1 = get_dataset(get_normal_vector(2, N), M1, b2_1, N)
    X2_2 = get_dataset(get_normal_vector(2, N), M2, b2_2, N)
    X2_3 = get_dataset(get_normal_vector(2, N), M3, b2_3, N)

    x0_12_vector, x1_12_a_vector, x1_12_b_vector = get_baios_line(GRAPH_SIZE, b2_1, b2_2, M1, M2, 1.0 / 3.0)

    for i in range (0, 1999):
        if x1_12_b_vector[i] > 1.0:
            x0_12_vector[i] = 0
            x1_12_a_vector[i] = 0
            x1_12_b_vector[i] = 0

    x0_13_vector, x1_13_a_vector, x1_13_b_vector = get_baios_line(GRAPH_SIZE, b2_1, b2_3, M1, M3, 1.0 / 3.0)

    for i in range (0, 1999):
        if x1_13_a_vector[i] > 1.0:
            x0_13_vector[i] = 0
            x1_13_a_vector[i] = 0
            x1_13_b_vector[i] = 0

    x0_23_vector, x1_23_a_vector, x1_23_b_vector = get_baios_line(GRAPH_SIZE, b2_2, b2_3, M2, M3, 1.0 / 3.0)

    for i in range (0, 1999):
        if x1_23_a_vector[i] < 1.0:
            x0_23_vector[i] = 0
            x1_23_a_vector[i] = 0
            x1_23_b_vector[i] = 0

    plt.figure(figsize=GRAPH_SIZE)
    plt.title("Baios for 3")
    plt.scatter(X2_1[0], X2_1[1], c='black')
    plt.scatter(X2_2[0], X2_2[1], c='pink')
    plt.scatter(X2_3[0], X2_3[1], c='red')
    plt.scatter(x0_12_vector, x1_12_a_vector, c='green')
    plt.scatter(x0_12_vector, x1_12_b_vector, c='blue')
    plt.scatter(x0_13_vector, x1_13_a_vector, c='orange')
    plt.scatter(x0_13_vector, x1_13_b_vector, c='purple')
    plt.scatter(x0_23_vector, x1_23_a_vector, c='lime')
    plt.scatter(x0_23_vector, x1_23_b_vector, c='yellow')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()

    X_to_test = get_dataset(get_normal_vector(2, N * 20), M1, b2_1, N * 20)
    print(f'Относительная экспериментальная вероятность: {get_experimental_probability(0.5, N)}')

    p_max = 0.05
    p = 1.0
    i = 0
    sum = 0

    while p >= p_max or i < 200:
        sum += baios_classifier(0.5, 0.5, b2_1, b2_2, M1, M2, np.array([X_to_test[0][i], X_to_test[1][i]]))
        i += 1
        p = sum / i
    print(f'Объем обучающей выборки: {i}')
