import numpy as np
from matplotlib import pyplot as plt

from utils.consts import COLORS
from utils.utils import get_euclid_dist, show_vector_points_v1, get_s, show_samples_v0, concatenate_samples, \
    get_sum_distance, merge_classes


def classify_data(train_samples, test_samples, classifier, params):

    entire_results = []
    for test_sample in test_samples:
        intermediate_results = []

        for j in range(test_sample.shape[1]):
            classification_result = classifier(
                test_sample[:, j],
                train_samples,
                params)
            intermediate_results.append(classification_result)
        entire_results.append(np.array(intermediate_results))
    return np.array(entire_results)


def get_centres(samples):

    m0, m0_max_dist     = find_first_center_and_max_distance(samples)
    m1, m1_max_dist     = find_second_center_and_max_distance(samples, m0)
    centers = [m0, m1]
    remaining_samples   = remove_center_from_samples(samples, m0)
    remaining_samples   = remove_center_from_samples(remaining_samples, m1)

    max_distance_array  = [m0_max_dist, m1_max_dist]
    t_distance_array    = [0, 0]
    colors              = ['red', 'green', 'blue', 'yellow', 'pink']
    counter             = 0

    while True:
        distances_array = get_distances_to_centers(remaining_samples, centers)
        min_indexes     = np.argmin(distances_array, axis=0)
        counter += 1
        if counter >= 5:
            break

        show_samples_with_min_indexes(remaining_samples, min_indexes, colors, centers)
        plt.show()

        distances_array_min             = get_min_distances(min_indexes, distances_array)
        max_indexes                     = np.argmax(distances_array_min)
        centers_candidate               = remaining_samples[0:2, max_indexes]
        typical_distance                = get_typical_distance(np.array(centers))
        tmp_center_candidate            = centers_candidate.reshape(2, 1)
        distances_array_from_centers    = get_distances_to_centers(tmp_center_candidate, centers)
        d_min_index                     = np.argmin(distances_array_from_centers)
        d_min                           = distances_array_from_centers[d_min_index]

        max_distance_array.append(distances_array_min[max_indexes])
        t_distance_array.append(typical_distance)

        if d_min < typical_distance:
            break

        centers.append(centers_candidate)
        remaining_samples = remove_center_from_samples(samples, centers_candidate)
    return centers, max_distance_array, t_distance_array


def find_first_center_and_max_distance(samples):

    samples_mean = np.mean(samples, axis=1)
    samples_mean = samples_mean.reshape(2, 1)
    distances = []
    for i in range(0, len(samples[1])):
        distances.append(get_euclid_dist(samples_mean, samples[0:2, i]))
    m0 = np.argmax(distances)
    return samples[0:2, m0], distances[m0]


def find_second_center_and_max_distance(samples, m0):

    distances = []
    for i in range(0, len(samples[1])):
        distances.append(get_euclid_dist(m0, samples[0:2, i]))
    m1 = np.argmax(distances)
    return samples[0:2, m1], distances[m1]


def remove_center_from_samples(samples, center):

    center = center.reshape(2, 1)
    copy = samples.copy()
    del_index = np.where(copy[0:2, :] == center)
    return np.delete(copy, del_index[1][1], axis=1)


def get_distances_to_centers(samples, centers_array):

    N = len(samples[1])
    distances   = np.zeros(shape=(len(centers_array), N))

    for i in range(0, len(centers_array)):

        center  = centers_array[i]
        for j in range(0, N):
            distances[i, j] = get_euclid_dist(center, samples[0:2, j])
    return distances


def get_min_distances(min_indexes, distances_array):

    res = []
    for i in range(0, len(min_indexes)):
        res.append(distances_array[min_indexes[i], i])
    return np.array(res)


def get_typical_distance(centers_array):

    center_distances = []
    i = 0

    for center in centers_array:
        center = center.reshape(2, 1)
        center_distances.append(get_distances_to_centers(center, centers_array))
        i += 1

    sum_distance = get_sum_distance(np.triu(center_distances))
    return 0.5 * sum_distance / len(center_distances)


def show_samples_with_min_indexes(samples, min_indexes, colors, centers):
    classes = [[] for _ in range(0, len(centers))]
    for i, cls in np.ndenumerate(min_indexes):
        classes[cls].append(samples[0:2, i])

    for k in range(0, len(centers)):
        classes[k]  = np.array(classes[k])
    for class_, color in zip(classes, colors):
        shape       = class_.shape
        new_shape   = (shape[0], shape[1])
        class_      = np.reshape(class_, new_shape)
        class_      = np.transpose(class_)
        show_vector_points_v1(class_, color)

    plt.title(f'Кол-во центров = {len(centers)}')
    for c in centers:
        plt.scatter(c[0], c[1], marker='o', color='black', alpha=0.6, s=100)


def show_classes(classes, colors, centers):

    for class_, color in zip(classes, colors):
        shape   = class_.shape
        new_shape = (shape[0], shape[1])
        class_ = np.reshape(class_, new_shape)
        class_ = np.transpose(class_)
        show_vector_points_v1(class_, color)

    for c in centers:
        plt.scatter(c[0], c[1], marker='o', color='black', alpha=0.6, s=100)


def k_means_method(samples, K, indexes_clusters=None):

    centers = []

    if indexes_clusters is not None:
        for index in indexes_clusters:
            centers.append(samples[0:2, index])
    else:
        for k in range(0, K):
            centers.append(samples[0:2, k])

    iter = 0
    stats_num_changed = []
    classes = None
    colors = ['red', 'green', 'blue', 'yellow', 'pink']

    while True:

        old_classes = classes
        distances   = get_distances_to_centers(samples, centers)
        min_indexes = np.argmin(distances, axis=0)
        classes     = get_s(samples, min_indexes, K)

        show_classes(classes, colors, centers)

        plt.title(f'Номер итерации: {iter}')
        plt.show()

        iter += 1
        changed_flag = False

        if old_classes is not None:

            cur_changed = 0

            for i in range(K):
                for j in range(len(classes[i])):
                    if classes[i][j] not in old_classes[i]:
                        cur_changed += 1

            stats_num_changed.append(cur_changed)

        for i in range(K):
            new_center = classes[i].mean(axis=0)

            if not np.array_equal(new_center, centers[i]):
                changed_flag = True
            centers[i] = new_center

        if not changed_flag:
            break

    return centers, classes, stats_num_changed


def calc_and_show_k_means(merged_train_samples, K):

    indexes = rng.choice(range(merged_train_samples.shape[1]), K, replace=False)
    centers, classes, stats = k_means_method(merged_train_samples, K, indexes)
    plt.title(f'k средних для {K} классов')

    for k in range(len(classes)):
        plt.scatter(classes[k][:, 0], classes[k][:, 1], label=f"cl{k}")
    for c in centers:
        plt.scatter(c[0], c[1], marker='o', color='black', alpha=0.6, s=100)

    x = np.arange(3, 3 + len(stats))

    plt.show()
    plt.title(f'k средних для {K} классов')
    plt.plot(x, stats, label='Зависимость количества изменений от номера итерации')
    plt.xlabel('Количество итераций')
    plt.ylabel('Количество измененных векторов')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # 1. Смоделировать и изобразить графически обучающие выборки объема N = 50 для пяти
    # нормально распределенных двумерных случайных векторов с заданными математическими
    # ожиданиями и самостоятельно подобранными корреляционными матрицами, которые обеспечивают
    # линейную разделимость классов.

    train_samples0  = np.load('resources/datasets/red_dataset.npy').T
    train_samples1  = np.load('resources/datasets/blue_dataset.npy').T
    train_samples2  = np.load('resources/datasets/green_dataset.npy').T
    train_samples3  = np.load('resources/datasets/yellow_dataset.npy').T
    train_samples4  = np.load('resources/datasets/purple_dataset.npy').T
    vectors         = [train_samples0, train_samples1, train_samples2, train_samples3, train_samples4]
    show_samples_v0(vectors, COLORS)
    plt.show()

    # 2. Объединить пять выборок в одну. Общее количество векторов в объединенной выборке должно быть 250.
    # Полученная объединенная выборка используется для выполнения пунктов 3 и 4 настоящего плана.

    merged_train_samples = merge_classes(vectors)

    # 3. Разработать программу кластеризации данных с использованием минимаксного алгоритма.
    # В качестве типичного расстояния взять половину среднего расстояния между существующими кластерами.
    # Построить отображение результатов кластеризации для числа кластеров, начиная с двух.
    # Построить график зависимости максимального (из минимальных) и типичного расстояний от числа кластеров.

    plt.title(f'Минимакс для 5-ти классов')

    res = merged_train_samples
    m_array, max_distances, typical_distances = get_centres(res)

    for j in range(0, 5):
        show_vector_points_v1(vectors[j], COLORS[j])
    for m in m_array:
        plt.scatter(m[0], m[1], marker='o', color='black', alpha=0.6, s=100)
    plt.show()

    x = np.arange(0, len(m_array))
    plt.title(f'Минимакс для 5-ти классов')
    plt.plot(x, max_distances, label='Максимальное расстояние')
    plt.plot(x, typical_distances, label='Типичное расстояние')
    plt.xlabel('Кол-во центров')
    plt.legend()
    plt.show()

    # 4. Разработать программу кластеризации данных с использованием алгоритма К внутригрупповых
    # средних для числа кластеров равного 3 и 5. Для ситуации 5 кластеров подобрать начальные
    # условия так, чтобы получить два результата: а) чтобы кластеризация максимально соответствовал
    # первоначальному разбиению на классы (“правильная” кластеризация); б) чтобы кластеризация
    # максимально не соответствовала первоначальному разбиению на классы (“неправильная” кластеризация).
    # Для всех случаев построить графики зависимости числа векторов признаков, сменивших номер кластера,
    # от номера итерации алгоритма.

    K = 3
    rng = np.random.default_rng(2)
    calc_and_show_k_means(merged_train_samples, K)

    rng_array = [np.random.default_rng(42), np.random.default_rng(1)]
    for rng in rng_array:
        K = 5
        samples_array_result = concatenate_samples(vectors)
        calc_and_show_k_means(merged_train_samples, K)
