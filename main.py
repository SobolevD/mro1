import numpy as np

from utils.K_neighbours import K_neighbours_classifier
from utils.consts import M0, B0, M1, B1
from utils.errors import classification_error_by_params
from utils.normal import get_dataset, get_normal_vector
from utils.parzen import parzen_classifier


def classify_data(train_datasets, test_datasets, classifier, params):

    entire_results = []
    for test_sample in test_datasets:
        intermediate_results = []

        for j in range(test_sample.shape[1]):

            classification_result = classifier(
                test_sample[:, j],
                train_datasets,
                params)

            intermediate_results.append(classification_result)

        entire_results.append(np.array(intermediate_results))

    return np.array(entire_results)


def get_classification_error_from_class_nums(y_pred, y_test):

    N0 = np.sum(y_test == 0)
    N1 = np.sum(y_test == 1)

    y_pred = np.ravel(y_pred)

    p0_count = 0
    p1_count = 0

    for test_class_num, pred_class_num in zip(y_test, y_pred):

        if [test_class_num, pred_class_num] == [0, 1]:
            p0_count += 1

        if [test_class_num, pred_class_num] == [1, 0]:
            p1_count += 1

    return p0_count / N0, p1_count / N1


if __name__ == '__main__':

    print("===================================")
    print("========== Метод Парзена ==========")
    print("===================================")

    n = 2
    train_dataset_len   = 50
    test_dataset_len    = 100

    train_dataset0  = get_dataset(get_normal_vector(n, train_dataset_len), M0, B0, train_dataset_len)
    train_dataset1  = get_dataset(get_normal_vector(n, train_dataset_len), M1, B1, train_dataset_len)
    train_datasets  = [train_dataset0, train_dataset1]

    test_dataset0   = get_dataset(get_normal_vector(n, test_dataset_len), M0, B0, test_dataset_len)
    test_dataset1   = get_dataset(get_normal_vector(n, test_dataset_len), M1, B1, test_dataset_len)
    test_datasets   = [test_dataset0, test_dataset1]

    class_num       = np.concatenate((np.zeros(train_dataset_len), np.ones(train_dataset_len)), axis=0)
    test_class_num  = np.concatenate((np.zeros(test_dataset_len), np.ones(test_dataset_len)), axis=0)

    y_predicted     = classify_data(train_datasets, test_datasets, parzen_classifier, None)
    p0, p1          = get_classification_error_from_class_nums(y_predicted, test_class_num)
    empirical_risk  = 0.5 * p0 + 0.5 * p1

    print(f"Вероятность ошибочной классификации объекта класса 0 в класс 1: {p0}")
    print(f"Вероятность ошибочной классификации объекта класса 1 в класс 0: {p1}")
    print(f"Эмпирический риск: {empirical_risk}")

    K_array = [1, 3, 5]
    for K in K_array:

        print("===================================")
        print("==== Метод K ближайшик соседей ====")
        print(f"============= K = {K} =============")
        print("===================================")

        params_for_classifier   = [K, class_num]

        y_predicted             = classify_data(
            train_datasets,
            test_datasets,
            K_neighbours_classifier,
            params_for_classifier)

        p0, p1                  = get_classification_error_from_class_nums(y_predicted, test_class_num)
        empirical_risk          = 0.5 * p0 + 0.5 * p1

        print(f"Вероятность ошибочной классификации объекта класса 0 в класс 1: {p0}")
        print(f"Вероятность ошибочной классификации объекта класса 1 в класс 0: {p1}")
        print(f"Эмпирический риск: {empirical_risk}")

    print("==================================")
    print("====== Классификатор Байеса ======")
    print("==================================")

    p0              = classification_error_by_params(test_dataset0, M0, M1, B0, B1, 0.5, 0.5)
    p1              = classification_error_by_params(test_dataset1, M1, M0, B1, B0, 0.5, 0.5)
    empirical_risk  = 0.5 * p0 + 0.5 * p1

    print(f"Вероятность ошибочной классификации объекта класса 0 в класс 1: {p0}")
    print(f"Вероятность ошибочной классификации объекта класса 1 в класс 0: {p1}")
    print(f"Эмпирический риск: {empirical_risk}")
