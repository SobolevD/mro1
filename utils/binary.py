import math
import numpy as np

from utils.laplas import laplas_function

VALUE_ONE = 1
VALUE_ZERO = 0

LETTER_C = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1]])

LETTER_P = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0]])

PROBABILITY_CLASS_C = 0.5
PROBABILITY_CLASS_P = 0.5


# =============================
# Основная работа с матрицами
# =============================
def invert_value(p, value, p_change=0.3):
    return value if p > p_change else 1 - value


def transform_matrices(letter_matrix, selection_size=200, p_change=0.3):

    letter_shape = np.shape(letter_matrix)
    matrix_as_vector = letter_matrix.flatten()
    result = np.zeros((selection_size, letter_shape[0] * letter_shape[1]))

    invert_matrix_values = np.vectorize(invert_value)
    for i in range(0, selection_size, 1):
        random_vector = np.random.uniform(0, 1, letter_shape[0] * letter_shape[1])
        result[i] = invert_matrix_values(random_vector, matrix_as_vector, p_change)

    # Если выборка одна, просто вернем один массив
    if selection_size == 1:
        return result[0].reshape(letter_matrix.shape)
    return result


def calc_condition_probs(set_changed_vectors):

    shape_of_array = np.shape(set_changed_vectors)  # 200 x 81
    shape_result_array = np.shape(set_changed_vectors[0])

    cum_sum = np.zeros(shape_result_array)
    for i in range(0, shape_of_array[0], 1):
        cum_sum += set_changed_vectors[i]

    print(f'CUMSUM: {cum_sum}')
    return np.divide(cum_sum, shape_of_array[0])


# =============================
# Параметры бинарного распределения
# =============================
def calc_binarySD(cond_probs_array_0, cond_probs_array_1):
    size = cond_probs_array_0.size

    part_0 = np.zeros(size)
    part_1 = np.zeros(size)

    for i in range(0, size, 1):
        log_part = calc_Wlj_coef_arr(cond_probs_array_1, cond_probs_array_0)[i]
        part_0[i] = np.power(log_part, 2) * cond_probs_array_0[i] * (1 - cond_probs_array_0[i])
        part_1[i] = np.power(log_part, 2) * cond_probs_array_1[i] * (1 - cond_probs_array_1[i])

    return np.sqrt(np.sum(part_0)), np.sqrt(np.sum(part_1))


def calc_M(cond_probs_array_0, cond_probs_array_1):
    size = cond_probs_array_0.size

    m0_part = np.zeros(size)
    m1_part = np.zeros(size)

    wlj_array = calc_Wlj_coef_arr(cond_probs_array_1,
                                  cond_probs_array_0)
    for i in range(0, size, 1):
        m0_part[i] = wlj_array[i] * cond_probs_array_0[i]

        m1_part[i] = wlj_array[i] * cond_probs_array_1[i]

    return np.sum(m0_part), np.sum(m1_part)


# =============================
# Байесовский классификатор (обычный и для массива)
# =============================
def classify_vectors_array(vectors_array, Pl, Pj, cond_probs_array_l, cond_pobs_array_j):
    shape   = np.shape(vectors_array)
    result  = np.zeros(shape[0], int)

    for i in range(0, shape[0], 1):
        result[i] = classify_Bayes(vectors_array[i], Pl, Pj, cond_probs_array_l, cond_pobs_array_j)

    return result


def classify_Bayes(X_v, Pl, Pj, cond_probs_array_l, cond_probs_array_j):
    Lamda_tilda  = calc_big_lambda(X_v, cond_probs_array_l, cond_probs_array_j)
    lambda_tilda = calc_small_lambda(Pl, Pj, cond_probs_array_l, cond_probs_array_j)
    return int(0) if Lamda_tilda >= lambda_tilda else int(1)


# =============================
# Подсчет ошибок
# =============================
def calculate_exp_error(classified_array):
    return float(np.sum(classified_array) / classified_array.size)


def calc_theoretical_error(p0, p1, cond_probs_array_0, cond_probs_array_1):
    sm_lambda   = calc_small_lambda(p0, p1, cond_probs_array_0, cond_probs_array_1)

    m0, m1      = calc_M(cond_probs_array_0, cond_probs_array_1)
    sd0, sd1    = calc_binarySD(cond_probs_array_0, cond_probs_array_1)

    p0          = 1 - laplas_function((sm_lambda - m0) / sd0)
    p1          = laplas_function((sm_lambda - m1) / sd1)

    return np.array([p0, p1])


# =============================
# Переменные промежуточных расчетов
# =============================
def calc_Wlj_coef_arr(cond_probs_array_l, cond_probs_array_j):
    shape = np.shape(cond_probs_array_l)
    result = np.zeros(shape)

    for i in range(0, shape[0], 1):
        result[i] = math.log(
            ((cond_probs_array_l[i] / (1 - cond_probs_array_l[i])) *
             ((1 - cond_probs_array_j[i]) / cond_probs_array_j[i]))
        )
    return result


def calc_small_lambda(Pl, Pj, cond_probs_array_l, cond_probs_array_j):
    shape = np.shape(cond_probs_array_l)
    result_part = np.zeros(shape)

    for i in range(0, shape[0], 1):
        result_part[i] = math.log((1 - cond_probs_array_l[i]) / (1 - cond_probs_array_j[i]))

    return math.log(Pj / Pl) + np.sum(result_part)


def calc_big_lambda(X_vector, cond_probs_array_l, cond_probs_array_j):
    array_wlj = calc_Wlj_coef_arr(cond_probs_array_l, cond_probs_array_j)
    return np.sum(X_vector * array_wlj)