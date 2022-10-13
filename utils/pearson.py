from math import exp

import numpy as np

from utils.baios import __mul_3_matrices
from utils.laplas import inv_laplas_function, get_mahalanobis_distance


def Neyman_Pearson(Bj, Bl, Mj, Ml, Pl, Pj, k):
    d, e = __get_de(Mj, Ml, Bj, Bl)
    f = __get_f(Mj, Ml, Bj, Pl, Pj)

    return -1 / d * (e * k + f)


def __get_a(Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    return Bj_inv[0][0] - Bl_inv[0][0]


def __get_b(Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    return (Bj_inv[1][0] - Bl_inv[1][0]) + (Bj_inv[0][1] - Bl_inv[0][1])


def __get_c(Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    return Bj_inv[1][1] - Bl_inv[1][1]


def __get_de(Mj, Ml, Bj, Bl):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    Mj_transpos = np.transpose(Mj)
    Ml_transpos = np.transpose(Ml)

    Ml_mul_Bl = np.matmul(Ml_transpos, Bl_inv)
    Mj_mul_Bj = np.matmul(Mj_transpos, Bj_inv)

    res = 2 * (Ml_mul_Bl - Mj_mul_Bj)
    return res[0], res[1]


def __get_f(Mj, Ml, B, pl, pj):
    # Слагаемое 1
    sum_part1 = np.log(get_lambda(Mj, Ml, B, pl))

    # Слагаемое 2
    sum_part2 = 2 * np.log(pl / pj)

    # Слагаемое 3
    B_inv = np.linalg.inv(B)
    Mj_transpos = np.transpose(Mj)
    Ml_transpos = np.transpose(Ml)

    MlBlMl = __mul_3_matrices(Ml_transpos, B_inv, Ml)
    MjBjMj = __mul_3_matrices(Mj_transpos, B_inv, Mj)

    return sum_part1 + sum_part2 - MlBlMl + MjBjMj

def get_neuman_pearson_line(shape, b, m1, m2, p):

    p1 = 1 - p

    step = 0.005
    points_count = shape[0] / step

    x0_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))
    x1_a_vector = np.reshape(np.zeros(int(points_count)), (int(points_count), 1))

    counter = 0
    i = -(shape[0] / 2.0)
    while i < shape[1] / 2.0 - step:
        x0_vector[counter] = i
        x1_vector = Neyman_Pearson(b, b, m1, m2, p, p1, i)
        x1_a_vector[counter] = x1_vector
        counter += 1
        i += step
    return x0_vector, x1_a_vector


def get_lambda(M1, M2, B, p):
    mh_distance = get_mahalanobis_distance(M1, M2, B)
    return exp(-0.5 * mh_distance + np.sqrt(mh_distance) * inv_laplas_function(1 - p))
