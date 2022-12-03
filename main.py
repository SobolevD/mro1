import numpy as np
from scipy.spatial import distance

from utils.consts import M0, B0, M1, B1
from utils.errors import classification_error_by_params
from utils.normal import get_dataset, get_normal_vector
from utils.parzen import parzen_classifier


def get_distances(test_dataset_element, datasets, class_nums):

    distances_with_classes = []

    count = 0
    for train_dataset in datasets:
        train_elements_count = train_dataset.shape[1]
        for element_num in range(train_elements_count):
            distances_with_classes.append(
                (distance.euclidean(test_dataset_element, train_dataset[:, element_num]), class_nums[count])
            )
            count += 1

    return distances_with_classes


def K_neighbours_classifier(x, datasets, params):

    K = params[0]
    class_nums = params[1]

    distances = get_distances(x, datasets, class_nums)
    distances.sort(key=lambda row: row[0])
    distances = np.array(distances)

    K_0 = 0
    K_1 = 0

    for i in range(0, K):
        if distances[i, 1] == 0:
            K_0 += 1
            continue
        K_1 += 1

    return 0 if K_0 > K_1 else 1


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

    train_dataset0 = get_dataset(get_normal_vector(2, 50), M0, B0, 50)
    train_dataset1 = get_dataset(get_normal_vector(2, 50), M1, B1, 50)
    train_datasets = [train_dataset0, train_dataset1]

    test_dataset0 = get_dataset(get_normal_vector(2, 100), M0, B0, 100)
    test_dataset1 = get_dataset(get_normal_vector(2, 100), M1, B1, 100)
    test_datasets = [test_dataset0, test_dataset1]

    # print("\nклассификатор Байеса, 2 класса")
    p0 = classification_error_by_params(test_dataset0, M0, M1, B0, B1, 0.5, 0.5)
    p1 = classification_error_by_params(test_dataset1, M1, M0, B1, B0, 0.5, 0.5)
    # print(f"Вероятность ошибочной классификации p01: {p0}")
    # print(f"Вероятность ошибочной классификации p10: {p1}")
    # print(f"Эмпирический риск: {0.5 * p0 + 0.5 * p1}")

    y_test = np.zeros(100)
    y_test = np.concatenate((y_test, np.ones(100)), axis=0)
    y_pred = classify_data(train_datasets, test_datasets, parzen_classifier, None)
    p0, p1 = get_classification_error_from_class_nums(y_pred, y_test)

    print("\nметод Парзена, 2 класса")
    print(f"Вероятность ошибочной классификации p01: {p0}")
    print(f"Вероятность ошибочной классификации p10: {p1}")
    print(f"Эмпирический риск: {0.5 * p0 + 0.5 * p1}")

    class_num   = np.concatenate((np.zeros(50), np.ones(50)), axis=0)
    y_test      = np.concatenate((np.zeros(100), np.ones(100)), axis=0)

    K_array = np.arange(1, 7, 2)
    for K in K_array:
        params = [K, class_num]
        y_pred = classify_data(train_datasets, test_datasets, K_neighbours_classifier, params)
        p0, p1 = get_classification_error_from_class_nums(y_pred, y_test)
        print(f"\nметод K ближайших (K = {K}) соседей 2 класса")
        print(f"Вероятность ошибочной классификации p01: {p0}")
        print(f"Вероятность ошибочной классификации p10: {p1}")
        print(f"Эмпирический риск: {0.5 * p0 + 0.5 * p1}")