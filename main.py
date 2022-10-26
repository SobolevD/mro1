import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils.binary import LETTER_C, LETTER_P, transform_matrix, transform_matrices, \
    calculate_array_of_condition_probabilities, calculate_binary_SD, calculate_binary_m, calculate_lambda_tilda, \
    classify_array_of_vectors, PROBABILITY_CLASS_C, PROBABILITY_CLASS_P, calculate_exp_error, \
    calculate_theoretical_errors

if __name__ == '__main__':

    result_1 = transform_matrix(LETTER_C, 0.3)
    result_2 = transform_matrix(LETTER_P, 0.3)

    figure = plt.figure(figsize=(10, 10))
    plt.title("Letters")
    sub_figure_1 = figure.add_subplot(2, 2, 1)
    plt.imshow(1 - LETTER_C, cmap='gray')
    sub_figure_1.set_title("Буква П")

    sub_figure_2 = figure.add_subplot(2, 2, 2)
    plt.imshow(1 - result_1, cmap='gray')
    sub_figure_2.set_title("Буква П после обработки")

    sub_figure_3 = figure.add_subplot(2, 2, 3)
    plt.imshow(1 - LETTER_P, cmap='gray')
    sub_figure_3.set_title("Буква Б")

    sub_figure_4 = figure.add_subplot(2, 2, 4)
    plt.imshow(1 - result_2, cmap='gray')
    sub_figure_4.set_title("Буква Б после обработки")
    plt.show()

    test_data_class_p = transform_matrices(LETTER_C, 200, 0.3)
    test_data_class_b = transform_matrices(LETTER_P, 200, 0.3)

    cond_prob_array_class_p = calculate_array_of_condition_probabilities(test_data_class_p)
    cond_prob_array_class_b = calculate_array_of_condition_probabilities(test_data_class_b)

    sd0, sd1 = calculate_binary_SD(cond_prob_array_class_p, cond_prob_array_class_b)
    m0, m1 = calculate_binary_m(cond_prob_array_class_p, cond_prob_array_class_b)
    print("ME 0", m0)
    print("ME 1", m1)
    print("SD 0", sd0)
    print("SD 1", sd1)
    fig = plt.figure
    plt.title("X0 blue and X1 red")
    x0 = np.arange(-25, 25, 0.001)
    plt.plot(x0, norm.pdf(x0, m0, sd1), color='blue', linewidth=3)

    x1 = np.arange(-25, 25, 0.001)
    plt.plot(x1, norm.pdf(x1, m1, sd1), color='red', linewidth=3)
    lambda_tilda = calculate_lambda_tilda(0.5, 0.5, cond_prob_array_class_p, cond_prob_array_class_b)
    array_lambda_tilda = np.zeros(4) + lambda_tilda
    plt.plot(array_lambda_tilda, np.arange(0, 0.2, 0.05), color='black')
    print("lambda_tilda: ", lambda_tilda)
    plt.show()

    classified_array_class_p = classify_array_of_vectors(test_data_class_p, PROBABILITY_CLASS_C,
                                                                         PROBABILITY_CLASS_P,
                                                                         cond_prob_array_class_p,
                                                                         cond_prob_array_class_b)

    classified_array_class_b = classify_array_of_vectors(test_data_class_b, PROBABILITY_CLASS_C,
                                                                         PROBABILITY_CLASS_P,
                                                                         cond_prob_array_class_b,
                                                                         cond_prob_array_class_p)

    class_p_exp_error = calculate_exp_error(classified_array_class_p)
    class_b_exp_error = calculate_exp_error(classified_array_class_b)

    print("Экспериментальная ошибка классификации для класса П:", class_p_exp_error)
    print("Экспериментальная ошибка классификации для класса Б:", class_b_exp_error)

    theoretical_error = calculate_theoretical_errors(0.5, 0.5, cond_prob_array_class_p,
                                                                     cond_prob_array_class_b)

    print("Теоритическая ошибка классификации для класса П:", theoretical_error[0])
    print("Теоритическая ошибка классификации для класса Б:", theoretical_error[1])