import numpy as np
from matplotlib import pyplot as plt

def get_euclidean_distance(x, y):
    x = x.reshape(2, )
    y = y.reshape(2, )
    return np.linalg.norm(x - y)

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

def show_samples(samples_array, colors_array):
    for samples, colors in zip(samples_array, colors_array):
        plt.scatter(samples[0, :], samples[1, :], color=colors)


def concatenate_samples(samples_array):
    result = samples_array[0]
    for i in range(1, len(samples_array)):
        result = np.concatenate((result, samples_array[i]), axis=1)
    return result

def get_centres(samples):
    m0, m0_max_dist = find_first_center_and_max_distance(samples)
    m1, m1_max_dist = find_second_center_and_max_distance(samples, m0)
    centers_array = [m0, m1]
    remaining_samples = remove_center_from_samples(samples, m0)
    remaining_samples = remove_center_from_samples(remaining_samples, m1)
    max_distance_array = [m0_max_dist, m1_max_dist]
    t_distance_array = [0, 0]
    colors = ['red', 'green', 'blue', 'yellow', 'pink']
    while True:
        distances_array = get_distances_to_centers(remaining_samples, centers_array)
        min_indexes = np.argmin(distances_array, axis=0)
        show_samples_with_min_indexes(remaining_samples, min_indexes, colors, centers_array)
        plt.show()
        distances_array_min = get_min_distances(min_indexes, distances_array)
        max_indexes = np.argmax(distances_array_min)
        max_distance_array.append(distances_array_min[max_indexes])
        centers_candidate = remaining_samples[0:2, max_indexes]
        typical_distance = get_typical_distance(np.array(centers_array))
        t_distance_array.append(typical_distance)
        tmp_center_candidate = centers_candidate.reshape(2, 1)
        distances_array_from_centers = get_distances_to_centers(tmp_center_candidate, centers_array)
        d_min_index = np.argmin(distances_array_from_centers)
        d_min = distances_array_from_centers[d_min_index]
        if d_min < typical_distance:
            break
        centers_array.append(centers_candidate)
        remaining_samples = remove_center_from_samples(samples, centers_candidate)
    return centers_array, max_distance_array, t_distance_array
def find_first_center_and_max_distance(samples):
    samples_mean = np.mean(samples, axis=1)
    samples_mean = samples_mean.reshape(2, 1)
    distances = []
    for i in range(0, len(samples[1])):
        distances.append(get_euclidean_distance(samples_mean, samples[0:2, i]))
    m0 = np.argmax(distances)
    return samples[0:2, m0], distances[m0]


def find_second_center_and_max_distance(samples, m0):
    distances = []
    for i in range(0, len(samples[1])):
        distances.append(get_euclidean_distance(m0, samples[0:2, i]))
    m1 = np.argmax(distances)
    return samples[0:2, m1], distances[m1]


def remove_center_from_samples(samples, center):
    center = center.reshape(2, 1)
    copy = samples.copy()
    del_index = np.where(copy[0:2, :] == center)
    return np.delete(copy, del_index[1][1], axis=1)


def get_distances_to_centers(samples, centers_array):
    N = len(samples[1])
    distances = np.zeros(shape=(len(centers_array), N))
    for i in range(0, len(centers_array)):
        center = centers_array[i]
        for j in range(0, N):
            distances[i, j] = get_euclidean_distance(center, samples[0:2, j])
    return distances


def get_min_distances(min_indexes, distances_array):
    res = []
    for i in range(0, len(min_indexes)):
        res.append(distances_array[min_indexes[i], i])
    return np.array(res)


def get_sum_distance(distances_array):
    distance = 0
    for distances in distances_array:
        distance += sum(distances)
    return distance


def get_typical_distance(centers_array):
    center_distances = []
    i = 0
    for center in centers_array:
        center = center.reshape(2, 1)
        center_distances.append(get_distances_to_centers(center, centers_array))
        i += 1
    sum_distance = get_sum_distance(np.triu(center_distances))
    return 0.5 * sum_distance / len(center_distances)

def show_vector_points(X, color='red'):
    for x in X:
        plt.scatter(x[0], x[1], color=color)
def show_vector_points1(X, color='red'):
    plt.scatter(X[0, :], X[1, :], color=color)
def show_samples_with_min_indexes(samples, min_indexes, colors, centers):
    classes = [[] for _ in range(0, len(centers))]
    for i, cls in np.ndenumerate(min_indexes):
        classes[cls].append(samples[0:2, i])

    for k in range(0, len(centers)):
        classes[k] = np.array(classes[k])
    for class_, color in zip(classes, colors):
        shape = class_.shape
        new_shape = (shape[0], shape[1])
        class_ = np.reshape(class_, new_shape)
        class_ = np.transpose(class_)
        show_vector_points1(class_, color)

    plt.title(f'count centers = {len(centers)}')
    for c in centers:
        plt.scatter(c[0], c[1], marker='o', color='black', alpha=0.6, s=100)


colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "cyan",
    "magenta",
]
def get_s(samples, min_indexes, K):
    classes = [[] for _ in range(K)]
    for i, cls in np.ndenumerate(min_indexes):
        classes[cls].append(samples[0:2, i])

    for k in range(K):
        classes[k] = np.array(classes[k])

    return classes


def show_classes(classes, colors, centers):
    for class_, color in zip(classes, colors):
        shape = class_.shape
        new_shape = (shape[0], shape[1])
        class_ = np.reshape(class_, new_shape)
        class_ = np.transpose(class_)
        show_vector_points1(class_, color)
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
        distances = get_distances_to_centers(samples, centers)
        min_indexes = np.argmin(distances, axis=0)
        classes = get_s(samples, min_indexes, K)
        show_classes(classes, colors, centers)
        plt.title(f'iteration number: {iter}')
        iter += 1
        plt.show()
        changed = False

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
                changed = True
            centers[i] = new_center

        if not changed:
            break
    return centers, classes, stats_num_changed


def merge_classes(datasets):
    result = datasets[0]
    for i in range(1, len(datasets)):
        result = np.concatenate((result, datasets[i]), axis=1)
    return result

if __name__ == '__main__':

    # 1. Смоделировать и изобразить графически обучающие выборки объема N=50 для пяти
    # нормально распределенных двумерных случайных векторов с заданными математическими
    # ожиданиями и самостоятельно подобранными корреляционными матрицами, которые обеспечивают
    # линейную разделимость классов.
    N = 50
    #train_dataset0 = get_dataset(get_normal_vector(2, N), M0, B0, N).T
    #train_dataset1 = get_dataset(get_normal_vector(2, N), M1, B1, N).T
    #train_dataset2 = get_dataset(get_normal_vector(2, N), M2, B2, N).T
    #train_dataset3 = get_dataset(get_normal_vector(2, N), M3, B3, N).T
    #train_dataset4 = get_dataset(get_normal_vector(2, N), M4, B4, N).T

    train_dataset0 = np.load('resources/datasets/red_dataset.npy').T
    train_dataset1 = np.load('resources/datasets/blue_dataset.npy').T
    train_dataset2 = np.load('resources/datasets/green_dataset.npy').T
    train_dataset3 = np.load('resources/datasets/yellow_dataset.npy').T
    train_dataset4 = np.load('resources/datasets/purple_dataset.npy').T
    vectors = [train_dataset0, train_dataset1, train_dataset2, train_dataset3, train_dataset4]
    show_samples(vectors, colors)
    plt.show()

    a=5


    # 2. Объединить пять выборок в одну. Общее количество векторов в объединенной выборке должно быть 250.
    # Полученная объединенная выборка используется для выполнения пунктов 3 и 4 настоящего плана.
    merged_train_dataset = merge_classes(vectors)


    # 3. Разработать программу кластеризации данных с использованием минимаксного алгоритма.
    # В качестве типичного расстояния взять половину среднего расстояния между существующими кластерами.
    # Построить отображение результатов кластеризации для числа кластеров, начиная с двух.
    # Построить график зависимости максимального (из минимальных) и типичного расстояний от числа кластеров.
    plt.title(f'minmax for 5 classes')
    res = merged_train_dataset
    m_array, max_dist_array, t_dist_array = get_centres(res)
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)
    # print(m_array)
    for j in range(0, 5):
        show_vector_points1(vectors[j], colors[j])
    for m in m_array:

        plt.scatter(m[0], m[1], marker='o', color='black', alpha=0.6, s=100)


    fig.add_subplot(1, 2, 2)
    plt.title(f'minmax for 5 classes')
    x = np.arange(0, len(m_array) + 1)
    plt.plot(x, max_dist_array, label='max distance')
    plt.plot(x, t_dist_array, label='typical distance')
    plt.xlabel('count centers')
    plt.legend()
    plt.show()
    plt.show()
    a =5
    # 4. Разработать программу кластеризации данных с использованием алгоритма К внутригрупповых
    # средних для числа кластеров равного 3 и 5. Для ситуации 5 кластеров подобрать начальные
    # условия так, чтобы получить два результата: а) чтобы кластеризация максимально соответствовал
    # первоначальному разбиению на классы (“правильная” кластеризация); б) чтобы кластеризация
    # максимально не соответствовала первоначальному разбиению на классы (“неправильная” кластеризация).
    # Для всех случаев построить графики зависимости числа векторов признаков, сменивших номер кластера,
    # от номера итерации алгоритма.
    rng = np.random.default_rng(2)
    K = 3
    indexes = rng.choice(range(merged_train_dataset.shape[1]), K, replace=False)
    centers, classes, stats = k_means_method(merged_train_dataset, K, indexes)
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)
    plt.title(f'k means for {K} classes')
    for k in range(len(classes)):
        plt.scatter(classes[k][:, 0], classes[k][:, 1], label=f"cl{k}")
    for c in centers:
        plt.scatter(c[0], c[1], marker='o', color='black', alpha=0.6, s=100)
    fig.add_subplot(1, 2, 2)
    plt.title(f'k means for {K} classes')
    x = np.arange(3, 3 + len(stats))
    plt.plot(x, stats, label='dependence of the number of changes on the iteration number')
    plt.xlabel('count iteration')
    plt.ylabel('count changed vectors')
    plt.legend()
    plt.show()

    rng_array = [np.random.default_rng(42), np.random.default_rng(1)]
    for rng in rng_array:
        samples_array_result = concatenate_samples(vectors)
        K = 5
        indexes = rng.choice(range(samples_array_result.shape[1]), K, replace=False)
        centers, classes, stats = k_means_method(samples_array_result, K, indexes)
        fig = plt.figure(figsize=(15, 5))
        fig.add_subplot(1, 2, 1)
        plt.title(f'k means for {K} classes')
        for k in range(len(classes)):
            plt.scatter(classes[k][:, 0], classes[k][:, 1], label=f"cl{k}")
        for c in centers:
            plt.scatter(c[0], c[1], marker='o', color='black', alpha=0.6, s=100)
        fig.add_subplot(1, 2, 2)
        plt.title(f'k means for {K} classes')
        x = np.arange(3, 3 + len(stats))
        plt.plot(x, stats, label='dependence of the number of changes on the iteration number')
        plt.xlabel('count iteration')
        plt.ylabel('count changed vectors')
        plt.legend()
        plt.show()
    # Результат работы для 3-х кластеров

    # Результат работы для 5-х кластеров (“неправильная” кластеризация)
