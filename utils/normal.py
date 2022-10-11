import numpy as np

def get_normal_vector(dim, length):
    vector = np.zeros([dim, length], "uint8")

    for step in range(1, length):
        vector = vector + [np.random.uniform(-0.5, 0.5, length), np.random.uniform(-0.5, 0.5, length)]

    return vector / (np.sqrt(length) * np.sqrt(1 / 12))