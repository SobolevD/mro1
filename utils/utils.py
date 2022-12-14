import numpy as np
from matplotlib import pyplot as plt


def show_samples_v0(samples_array, colors_array):
    for samples, colors in zip(samples_array, colors_array):
        plt.scatter(samples[0, :], samples[1, :], color=colors)


def concatenate_samples(samples_array):
    result = samples_array[0]
    for i in range(1, len(samples_array)):
        result = np.concatenate((result, samples_array[i]), axis=1)
    return result


def get_classification_error_from_class_nums(y_pred, y_test):

    N0          = np.sum(y_test == 0)
    N1          = np.sum(y_test == 1)
    p0_count    = 0
    p1_count    = 0
    y_pred      = np.ravel(y_pred)

    for test_class_num, pred_class_num in zip(y_test, y_pred):

        if [test_class_num, pred_class_num] == [0, 1]:
            p0_count += 1

        if [test_class_num, pred_class_num] == [1, 0]:
            p1_count += 1

    return p0_count / N0, p1_count / N1


def get_euclid_dist(x, y):
    x = x.reshape(2, )
    y = y.reshape(2, )
    return np.linalg.norm(x - y)


def get_sum_distance(distances_array):
    distance = 0
    for distances in distances_array:
        distance += sum(distances)
    return distance


def show_vector_points(X, color='red'):
    for x in X:
        plt.scatter(x[0], x[1], color=color)


def show_vector_points_v1(X, color='red'):
    plt.scatter(X[0, :], X[1, :], color=color)


def get_s(samples, min_indexes, K):
    classes = [[] for _ in range(K)]
    for i, cls in np.ndenumerate(min_indexes):
        classes[cls].append(samples[0:2, i])

    for k in range(K):
        classes[k] = np.array(classes[k])

    return classes


def merge_classes(samples_set):
    result = samples_set[0]
    for i in range(1, len(samples_set)):
        result = np.concatenate((result, samples_set[i]), axis=1)
    return result
