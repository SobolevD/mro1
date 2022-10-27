import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils.binary import LETTER_C, LETTER_P, transform_matrices, \
    calc_condition_probs, calc_binarySD, calc_M, calc_small_lambda, \
    classify_vectors_array, calculate_exp_error, \
    calc_theoretical_error

if __name__ == '__main__':

    P_change = 0.3
    SET_SIZE = 200
    LETTER_SHAPE = LETTER_C.shape

    result_1 = transform_matrices(LETTER_C, 1)
    result_2 = transform_matrices(LETTER_P, 1)

    figure = plt.figure(figsize=(10, 10))
    plt.title("Letters")
    sub_figure_1 = figure.add_subplot(2, 2, 1)
    plt.imshow(1 - LETTER_C, cmap='gray')
    sub_figure_1.set_title("'Ц' letter")

    sub_figure_3 = figure.add_subplot(2, 2, 3)
    plt.imshow(1 - LETTER_P, cmap='gray')
    sub_figure_3.set_title("'P' letter")

    sub_figure_2 = figure.add_subplot(2, 2, 2)
    plt.imshow(1 - result_1, cmap='gray')
    sub_figure_2.set_title("Processed 'Ц' letter")

    sub_figure_4 = figure.add_subplot(2, 2, 4)
    plt.imshow(1 - result_2, cmap='gray')
    sub_figure_4.set_title("Processed 'P' letter")
    plt.show()

    test_data_class_c = transform_matrices(LETTER_C, SET_SIZE, P_change)
    test_data_class_p = transform_matrices(LETTER_P, SET_SIZE, P_change)

    cond_prob_array_class_c = calc_condition_probs(test_data_class_c)
    cond_prob_array_class_p = calc_condition_probs(test_data_class_p)

    sd0, sd1 = calc_binarySD(cond_prob_array_class_c, cond_prob_array_class_p)
    m0, m1 = calc_M(cond_prob_array_class_c, cond_prob_array_class_p)

    print("M0: ", m0)
    print("M1: ", m1)
    print("SD0: ", sd0)
    print("SD1: ", sd1)

    fig = plt.figure
    plt.title("Плотность вероятностей X0(с) X1(к)")
    x0 = np.arange(-25, 25, 0.001)
    plt.plot(x0, norm.pdf(x0, m0, sd0), color='blue', linewidth=3)

    x1 = np.arange(-25, 25, 0.001)
    plt.plot(x1, norm.pdf(x1, m1, sd1), color='red', linewidth=3)
    lambda_tilda = calc_small_lambda(0.5, 0.5, cond_prob_array_class_c, cond_prob_array_class_p)
    array_lambda_tilda = np.zeros(4) + lambda_tilda
    plt.plot(array_lambda_tilda, np.arange(0, 0.2, 0.05), color='black')
    print("lambda_tilda: ", lambda_tilda)
    plt.show()

    classified_array_class_c = classify_vectors_array(test_data_class_c,
                                                      0.5, 0.5,
                                                      cond_prob_array_class_c,
                                                      cond_prob_array_class_p)

    classified_array_class_p = classify_vectors_array(test_data_class_p,
                                                      0.5, 0.5,
                                                      cond_prob_array_class_p,
                                                      cond_prob_array_class_c)

    class_c_exp_error = calculate_exp_error(classified_array_class_c)
    class_p_exp_error = calculate_exp_error(classified_array_class_p)

    print("Экспериментальная ошибка классификации для класса Ц:", class_c_exp_error)
    print("Экспериментальная ошибка классификации для класса Р:", class_p_exp_error)

    theoretical_error = calc_theoretical_error(0.5, 0.5, cond_prob_array_class_c, cond_prob_array_class_p)

    print("Теоритическая ошибка классификации для класса Ц:", theoretical_error[0])
    print("Теоритическая ошибка классификации для класса Р:", theoretical_error[1])