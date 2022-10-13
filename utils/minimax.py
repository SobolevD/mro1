# Поиск априорной вероятности в общем случае для минимаксного классификатора (любая матрица штрафов)
from utils.baios import get_bayos_lines
import numpy as np


def minmax_get_p(c):
    p0 = (c[1][0] - c[1][1]) / (c[0][1] - c[0][0] + c[1][0] - c[1][1])
    return p0, 1 - p0


# Минимаксная решающая граница (частный случай Байесовского)
# @return - координата x
def get_minmax_border(b, c, m0, m1, y):
    p0, p1 = minmax_get_p(c)
    return get_bayos_lines(b, b, m0, m1, p0, p1, y)


def get_mn_line(shape, b, c, m1, m2):

    step = 0.005
    points_count = shape[0]/step

    x0_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))
    x1_a_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))

    counter = 0
    i = -(shape[0]/2.0)
    while i < shape[1]/2.0 - step:
        x0_vector[counter] = i
        x1_vector = get_minmax_border(b, c, m1, m2, i)
        x1_a_vector[counter] = x1_vector
        counter += 1
        i += step
    return x0_vector, x1_a_vector