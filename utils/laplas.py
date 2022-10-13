# интегральная функция нормального распределения через функцию ошибки
# erf = 2/sqrt(pi) integral [e^(-t^2)] dt for  0 to x
# F = 1/sqrt(2pi) integral [e^(-z^2/2)] dz for  0 to x
import math

from scipy.special import erf, erfinv

import numpy as np


def laplas_function(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


# обратная интегральная функция нормального распределения через обратную функцию ошибки
def inv_laplas_function(x):
    return np.sqrt(2) * erfinv(2 * x - 1)


def calculate_Mahalanobis_distance(M1, M2, cor_matrix_B):
    diference_M = np.reshape(M1, (1, 2)) - np.reshape(M2, (1, 2))
    return np.reshape(np.matmul(np.matmul(diference_M, cor_matrix_B), np.transpose(diference_M)), (1,))


# считаются теоретические вероятности ошибочной классификации

def get_p_error(M1, M2, cor_matrix_B, C, P0, P1):
    lambda_tilda = get_lambda(P0, P1, C)

    Mahalnobis_distance = calculate_Mahalanobis_distance(M1, M2, cor_matrix_B)

    # p01
    under_laplas_expression = (lambda_tilda + 0.5 * Mahalnobis_distance) / (np.sqrt(Mahalnobis_distance))
    p0 = 1 - laplas_function(under_laplas_expression)

    # p10
    under_laplas_expression = (lambda_tilda - 0.5 * Mahalnobis_distance) / (np.sqrt(Mahalnobis_distance))
    p1 = laplas_function(under_laplas_expression)
    return p0, p1


def get_lambda(P1, P2, C):
    return math.log(float((P1 * (C[0][1]) - C[0][0]) / (P2 * (C[1][0]) - C[1][1])))