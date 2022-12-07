import numpy as np
from matplotlib import pyplot as plt

from utils.consts import M0, B1, M1, B2, M2, B3, M3, B4, M4, B0
from utils.normal import get_dataset, get_normal_vector


def calc_dist_for_vectors(vectors, target_vector):
    return np.apply_along_axis(np.linalg.norm, 1, vectors - target_vector)


def calc_example_typical_distanse(centers):
    distances = np.array([])
    for center in centers:
        distances = np.append(distances, calc_dist_for_vectors(centers, center).ravel())
    distances = distances[distances != 0]
    return 0.5 * distances.sum() / len(distances)


def update_labels(x_samples, centers):
    curr_distances = np.array([])
    targets = np.array([])

    for i, sample in enumerate(x_samples):
        dist_cont = calc_dist_for_vectors(centers, sample)
        min_dist_center_index = np.argmin(dist_cont)
        curr_distances = np.append(curr_distances, dist_cont[min_dist_center_index])
        targets = np.append(targets, min_dist_center_index)

    return curr_distances, targets


def maximin_fit_predict(x_samples, typical_dist_func=None):
    def find_first_center(x_samples):
        mean_of_vectors = np.mean(x_samples, axis=0)
        first_center_index = np.argmax(
            calc_dist_for_vectors(x_samples_copy, mean_of_vectors)
        )
        return x_samples[first_center_index]

    def find_second_center(x_samples, first_center):
        second_center_index = np.argmax(calc_dist_for_vectors(x_samples, first_center))
        return x_samples[second_center_index]

    def update_centers(x_samples, curr_distances, centers):
        curr_distances_indices = np.argsort(curr_distances)
        candidate_index = curr_distances_indices[-1]
        candidate = x_samples[candidate_index]

        stats_maximin_dist.append(curr_distances[candidate_index])

        if curr_distances[candidate_index] > typical_dist_func(centers):
            centers = np.append(centers, [candidate], axis=0)

        return centers

    stats_typical_dist = []
    stats_maximin_dist = []
    stats_r = []
    r = 3

    x_samples_copy = x_samples.copy()
    targets = np.array([])
    centers = np.array([find_first_center(x_samples_copy)])

    second_center = find_second_center(x_samples_copy, centers[0])
    centers = np.append(centers, [second_center], axis=0)

    while True:
        update_flag = False
        curr_distances, targets = update_labels(x_samples, centers)
        centers_prev = centers.copy()

        stats_typical_dist.append(typical_dist_func(centers))
        stats_r.append(r)

        centers = update_centers(x_samples, curr_distances, centers)
        curr_distances, targets = update_labels(x_samples, centers)

        r += 1

        if np.array_equal(centers_prev, centers):
            break

    plt.title("Зависимость типичного расстояния от номера итерации")
    plt.plot(stats_r, stats_typical_dist)
    plt.show()
    plt.title("Зависимость максиминного расстояния от номера итерации")
    plt.plot(stats_r, stats_maximin_dist)
    plt.show()

    return x_samples_copy, targets, centers


def apply_algorithm(x_samples):

    x_samples, targets, centers = maximin_fit_predict(
        x_samples, calc_example_typical_distanse
    )
    res_clusters = {int(t): [] for t in set(targets)}
    for x, t in zip(x_samples, targets):
        res_clusters[t].append(x)

    return res_clusters, centers


def merge_classes(datasets, shuffle_mode=True):
    result = None
    for class_samples in datasets:
        if result is None:
            result = class_samples
            continue

        result = np.concatenate((result, class_samples), axis=0)
    if shuffle_mode:
        np.random.shuffle(result)
    return np.array(result)


def add_dataset_to_canvas(X, color):
    for x in X:
        x = np.array(x).flatten()
        plt.scatter(x[0], x[1], color=color)


def k_means_fit_predict(x_samples, K):
    def update_centers(x_samples, targets, centers):
        clusters = {int(t): [] for t in set(targets)}
        for x, t in zip(x_samples, targets):
            clusters[t].append(x)

        return np.array([np.mean(clusters[k], axis=0) for k in clusters])

    centers = x_samples[:K]
    targets = np.array([])

    r = 2
    stats = []

    while True:
        prev_targets = targets.copy()
        _, targets = update_labels(x_samples, centers)
        centers_prev = centers.copy()
        centers = update_centers(x_samples, targets, centers)

        equal = 1 if prev_targets.all() != targets.all() else 0

        stats.append((r, equal))
        if np.array_equal(centers_prev, centers):
            break
        r += 1

    return x_samples, targets, centers, np.array(stats)


colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "cyan",
    "magenta",
]


def show_result(res_clusters):
    for t in res_clusters:
        add_dataset_to_canvas(np.matrix(res_clusters[t]), colors[t])


if __name__ == '__main__':

    # 1. Смоделировать и изобразить графически обучающие выборки объема N=50 для пяти
    # нормально распределенных двумерных случайных векторов с заданными математическими
    # ожиданиями и самостоятельно подобранными корреляционными матрицами, которые обеспечивают
    # линейную разделимость классов.
    N = 50

    train_dataset0 = np.load('resources/datasets/red_dataset.npy')
    train_dataset1 = np.load('resources/datasets/blue_dataset.npy')
    train_dataset2 = np.load('resources/datasets/green_dataset.npy')
    train_dataset3 = np.load('resources/datasets/yellow_dataset.npy')
    train_dataset4 = np.load('resources/datasets/purple_dataset.npy')

    add_dataset_to_canvas(train_dataset0, "red")
    add_dataset_to_canvas(train_dataset1, "blue")
    add_dataset_to_canvas(train_dataset2, "green")
    add_dataset_to_canvas(train_dataset3, "yellow")
    add_dataset_to_canvas(train_dataset4, "purple")

    plt.show()

    # 2. Объединить пять выборок в одну. Общее количество векторов в объединенной выборке должно быть 250.
    # Полученная объединенная выборка используется для выполнения пунктов 3 и 4 настоящего плана.
    merged_train_dataset = merge_classes(
        [train_dataset0, train_dataset1, train_dataset2, train_dataset3, train_dataset4],
        False
    )

    add_dataset_to_canvas(merged_train_dataset, "purple")

    plt.show()

    # 3. Разработать программу кластеризации данных с использованием минимаксного алгоритма.
    # В качестве типичного расстояния взять половину среднего расстояния между существующими кластерами.
    # Построить отображение результатов кластеризации для числа кластеров, начиная с двух.
    # Построить график зависимости максимального (из минимальных) и типичного расстояний от числа кластеров.
    res_clusters, centers_2 = apply_algorithm(merge_classes([train_dataset0, train_dataset4]))

    add_dataset_to_canvas(train_dataset0, "red")
    add_dataset_to_canvas(train_dataset4, "purple")
    plt.scatter(centers_2[:2, 0], centers_2[:2, 1], color='black', s=100)
    plt.show()

    res_clusters, centers_3 = apply_algorithm(merge_classes([train_dataset0, train_dataset4, train_dataset3]))

    add_dataset_to_canvas(train_dataset0, "red")
    add_dataset_to_canvas(train_dataset4, "purple")
    add_dataset_to_canvas(train_dataset3, "yellow")
    plt.scatter(centers_3[:3, 0], centers_3[:3, 1], color='black', s=100)
    plt.show()

    res_clusters, centers_4 = apply_algorithm(merge_classes([train_dataset0, train_dataset4, train_dataset3, train_dataset2]))

    add_dataset_to_canvas(train_dataset0, "red")
    add_dataset_to_canvas(train_dataset4, "purple")
    add_dataset_to_canvas(train_dataset3, "yellow")
    add_dataset_to_canvas(train_dataset2, "green")
    plt.scatter(centers_4[:4, 0], centers_4[:4, 1], color='black', s=100)
    plt.show()

    res_clusters, centers_5 = apply_algorithm(merge_classes([train_dataset0, train_dataset4, train_dataset3, train_dataset2, train_dataset1]))

    add_dataset_to_canvas(train_dataset0, "red")
    add_dataset_to_canvas(train_dataset4, "purple")
    add_dataset_to_canvas(train_dataset3, "yellow")
    add_dataset_to_canvas(train_dataset2, "green")
    add_dataset_to_canvas(train_dataset1, "blue")
    plt.scatter(centers_5[:5, 0], centers_5[:5, 1], color='black', s=100)
    plt.show()

    plt.title("Зависимость макс. (из мин.) и типичного расстояний от числа кластеров", fontsize=10)
    plt.plot(
        list(range(2, 6)),
        [
            calc_example_typical_distanse(centers_2),
            calc_example_typical_distanse(centers_3),
            calc_example_typical_distanse(centers_4),
            calc_example_typical_distanse(centers_5),
        ],
    )
    plt.show()

    # 4. Разработать программу кластеризации данных с использованием алгоритма К внутригрупповых
    # средних для числа кластеров равного 3 и 5. Для ситуации 5 кластеров подобрать начальные
    # условия так, чтобы получить два результата: а) чтобы кластеризация максимально соответствовал
    # первоначальному разбиению на классы (“правильная” кластеризация); б) чтобы кластеризация
    # максимально не соответствовала первоначальному разбиению на классы (“неправильная” кластеризация).
    # Для всех случаев построить графики зависимости числа векторов признаков, сменивших номер кластера,
    # от номера итерации алгоритма.

    # Результат работы для 3-х кластеров
    x_samples = merge_classes([train_dataset4, train_dataset2, train_dataset3])
    x_samples, targets, centers, stats = k_means_fit_predict(x_samples, 3)

    res_clusters = {int(t): [] for t in set(targets)}
    for x, t in zip(x_samples, targets):
        res_clusters[t].append(x)

    for t in res_clusters:
        add_dataset_to_canvas(np.matrix(res_clusters[t]), colors[t])

    plt.scatter(centers[:, 0], centers[:, 1], color='black', s=100)
    plt.show()

    plt.title("Зависимость числа векторов признаков, сменивших номер кластера, от номера итерации алгоритма", fontsize=8)
    plt.plot(stats[:, 0], stats[:, 1])
    plt.show()

    # Результат работы для 5-х кластеров (“правильная” кластеризация)
    x_samples = merge_classes([train_dataset4, train_dataset2, train_dataset1, train_dataset3, train_dataset0])
    x_samples, targets, centers, stats = k_means_fit_predict(x_samples, 5)

    res_clusters = {int(t): [] for t in set(targets)}
    for x, t in zip(x_samples, targets):
        res_clusters[t].append(x)

    for t in res_clusters:
        add_dataset_to_canvas(np.matrix(res_clusters[t]), colors[t])

    plt.scatter(centers[:, 0], centers[:, 1], color='black', s=100)
    plt.show()

    plt.title("Зависимость числа векторов признаков, сменивших номер кластера, от номера итерации алгоритма", fontsize=8)
    plt.plot(stats[:, 0], stats[:, 1])
    plt.show()

    # Результат работы для 5-х кластеров (“неправильная” кластеризация)
    x_samples = merge_classes([train_dataset4, train_dataset2, train_dataset1, train_dataset3, train_dataset0])
    x_samples, targets, centers, stats = k_means_fit_predict(x_samples, 5)

    res_clusters = {int(t): [] for t in set(targets)}
    for x, t in zip(x_samples, targets):
        res_clusters[t].append(x)

    for t in res_clusters:
        add_dataset_to_canvas(np.matrix(res_clusters[t]), colors[t])

    plt.scatter(centers[:, 0], centers[:, 1], color='black', s=100)
    plt.show()

    plt.title("Зависимость числа векторов признаков, сменивших номер кластера, от номера итерации алгоритма", fontsize=8)
    plt.plot(stats[:, 0], stats[:, 1])
    plt.show()
