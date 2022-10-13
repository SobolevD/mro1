from matplotlib import pyplot as plt


def draw_for_two_vectors(title, size, X1, X2, line1, line2):
    plt.figure(figsize=size)
    plt.title(title)
    plt.scatter(X1[0], X1[1])
    plt.scatter(X2[0], X2[1])
    plt.scatter(line1, line2)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()
