import numpy as np

from utils.laplas import inv_laplas_function, get_mahalanobis_distance


def Neyman_Pearson(M0, M1, b, p, k):
    b01 = b[0][1]
    b00 = b[0][0]
    b10 = b[1][0]

    m0 = M0[0]
    m1 = M0[1]

    _m0 = M1[0]
    _m1 = M1[1]

    laplas = inv_laplas_function(1 - p)
    rho = get_mahalanobis_distance(M0, M1, b)

    x = (-k * (b01 * (-m0 + _m0) + b10 * (-m1 + _m1) - 1 / 2 * rho - np.sqrt(rho) * laplas)) / (
            b00 * (-m0 + _m0) + b10 * (-m1 + _m1))
    return x


def get_neuman_pearson_line(shape, b, m1, m2, p):
    step = 0.005
    points_count = shape[0] / step

    x0_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))
    x1_a_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))

    counter = 0
    i = -(shape[0] / 2.0)
    while i < shape[1] / 2.0 - step:
        x0_vector[counter] = i
        x1_vector = Neyman_Pearson(m1, m2, b, p, i)
        x1_a_vector[counter] = x1_vector[0]
        counter += 1
        i += step
    return x0_vector, x1_a_vector
