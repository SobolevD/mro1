import numpy as np
from scipy.spatial import distance


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
