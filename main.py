import matplotlib.pyplot as plt
import numpy as np
from qpsolvers import solve_qp
from sklearn.svm import SVC, LinearSVC


def plot(title: str, dataset0: np.array, dataset1: np.array, border_x_arr, border_y_arr, colors, labels):
    plt.figure()
    plt.title(title)

    plt.plot(dataset0[0, :, :], dataset0[1, :, :], color='red', marker='.')
    plt.plot(dataset1[0, :, :], dataset1[1, :, :], color='green', marker='+')

    for i in range(len(border_x_arr)):
        plt.plot(border_x_arr[i], border_y_arr[i], color=colors[i], label=labels[i])

    plt.legend()

    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)


def expand(x: np.array, value):
    return np.append(x, np.array([[value]]), axis=0)


def get_train_dataset(dataset1, dataset2):
    N = dataset1.shape[2] + dataset2.shape[2]  # 400 + 400 = 800
    train_dataset = []

    for i in range(N // 2):
        train_dataset.append(dataset1[:, :, i])
        # метка класса для функции r
        train_dataset[i * 2] = expand(train_dataset[i * 2], -1)

        train_dataset.append(dataset2[:, :, i])
        # метка класса для функции r
        train_dataset[i * 2 + 1] = expand(train_dataset[i * 2 + 1], 1)

    return np.array(train_dataset)


# ================================================================

def get_P(dataset, N):
    P = np.ndarray(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = np.matmul(dataset[j, :-1, :].T * r(dataset[j, :, :]),
                                dataset[i, :-1, :] * r(dataset[i, :, :]))
    return P


def r(x: np.array):
    return x[-1][0]

def get_A(dataset, N):
    A = np.zeros((N,))
    for j in range(N):
        A[j] = r(dataset[j])
    return A


def linear_discriminant(z: np.array, W: np.array) -> np.array:
    return np.matmul(W.T, z)

def linear_classificator(z: np.array, W: np.array):
    return 1 if linear_discriminant(z, W) > 0 else 0

def classification_error(dataset, W, class_id):
    errors = 0  # показывает число неверно определенных элементов
    N = dataset.shape[-1]

    for i in range(N):
        z = expand(dataset[:, :, i], 1)
        if linear_classificator(z, W) != class_id:
            errors += 1

    return errors / N  # ошибка первого рода


def linear_border(y: np.array, W: np.array):
    a = W[0]
    b = W[1]
    c = W[2]

    if a == 0.:
        return 0
    else:
        return (-b * y - c) / a

def task2(dataset1, dataset2):

    N = dataset1.shape[2] + dataset2.shape[2] # 100 + 100 = 200

    # подготовка обучающей выборки
    dataset = get_train_dataset(dataset1, dataset2)

    # параметры для решения задачи квадратичного программирования
    P = get_P(dataset, N)
    q = np.full((N, 1), -1, dtype=np.double)
    G = np.eye(N) * -1
    h = np.zeros((N,))
    A = get_A(dataset, N)
    b = np.zeros(1)
    eps = 1e-04

    # получаем вектор двойственных коэффициентов
    _lambda = solve_qp(P, q, G, h, A, b, solver='cvxopt')

    # опорные вектора для метода solve_qp
    support_vectors_positions = _lambda > eps
    support_vectors = dataset[support_vectors_positions, :-1, 0]
    support_vectors_classes = dataset[support_vectors_positions, -1, 0]
    red_support_vectors = support_vectors[support_vectors_classes == -1]
    green_support_vectors = support_vectors[support_vectors_classes == 1]

    # находим весовые коэффициенты из выражения через двойственные коэффициенты
    # и пороговое значение через весовые коэффициенты и опорные вектора
    W = np.matmul((_lambda * A)[support_vectors_positions].T, support_vectors).reshape(2, 1)
    w_N = np.mean(support_vectors_classes - np.matmul(W.T, support_vectors.T))
    W = expand(W, w_N)

    # обучение модели SVC (kernel=linear)
    svc_clf = SVC(C=(np.max(_lambda) + eps), kernel='linear')
    svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    # опорные вектора для метода SVC
    support_vectors_svc = svc_clf.support_vectors_
    support_vectors_svc_indices = svc_clf.support_
    support_vectors_svc_classes = dataset[support_vectors_svc_indices, -1, 0]
    red_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == -1]
    green_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == 1]

    # весовые коэффициенты и пороговое значение для модели SVC
    W_svc_clf = svc_clf.coef_.T
    w_N_svc_clf = svc_clf.intercept_[0]
    W_svc_clf = expand(W_svc_clf, w_N_svc_clf)

    # обучение модели LinearSVC
    linear_svc_clf = LinearSVC(C=(np.max(_lambda) + eps))
    linear_svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    # весовые коэффициенты и пороговое значение для модели LinearSVC
    W_linear_svc_clf = linear_svc_clf.coef_.T
    w_N_linear_svc_clf = linear_svc_clf.intercept_[0]
    W_linear_svc_clf = expand(W_linear_svc_clf, w_N_linear_svc_clf)

    # выводим весовые коэффициенты, полученные для каждого метода
    print(f"W:\n{W}\n"
          f"W_svc_clf:\n{W_svc_clf}\n"
          f"W_linear_svc_clf:\n{W_linear_svc_clf}\n")

    # строим разделяющую полосу и разделяющую гиперплоскость
    y = np.arange(-4, 4, 0.1)
    x = linear_border(y, W)
    x_svc_clf = linear_border(y, W_svc_clf)
    x_linear_svc_clf = linear_border(y, W_linear_svc_clf)

    plot(f"solve_qp (cvxopt)", dataset1, dataset2, [x, x + 1 / W[0], x - 1 / W[0]], [y, y, y],
         ['black', 'green', 'red'], ['', '', ''])
    plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1], color='red')
    plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1], color='green')

    plot(f"SVC(kernel=linear)", dataset1, dataset2,
         [x_svc_clf, x_svc_clf + 1 / W_svc_clf[0], x_svc_clf - 1 / W_svc_clf[0]], [y, y, y],
         ['black', 'green', 'red'], ['', '', ''])
    plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1], color='red')
    plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1], color='green')

    plot(f"LinearSVC", dataset1, dataset2,
         [x_linear_svc_clf, x_linear_svc_clf + 1 / W_linear_svc_clf[0], x_linear_svc_clf - 1 / W_linear_svc_clf[0]],
         [y, y, y],
         ['black', 'green', 'red'], ['', '', ''])
    plt.show()


def task3(dataset3, dataset4):
    N = dataset3.shape[2] + dataset4.shape[2] # 100 + 100 = 200

    # подготовка обучающей выборки
    dataset = get_train_dataset(dataset3, dataset4)

    # параметры для решения задачи квадратичного программирования
    P = get_P(dataset, N)
    q = np.full((N, 1), -1, dtype=np.double)
    G = np.concatenate((np.eye(N) * -1, np.eye(N)), axis=0)
    A = get_A(dataset, N)
    b = np.zeros(1)
    eps = 1e-04

    for C in [0.1, 1, 10, 20]:
        h = np.concatenate((np.zeros((N,)), np.full((N,), C)), axis=0)

        # получаем вектор двойственных коэффициентов
        _lambda = solve_qp(P, q, G, h, A, b, solver='cvxopt')

        # опорные вектора для метода solve_qp
        support_vectors_positions = _lambda > eps
        support_vectors = dataset[support_vectors_positions, :-1, 0]
        support_vectors_classes = dataset[support_vectors_positions, -1, 0]
        red_support_vectors = support_vectors[support_vectors_classes == -1]
        green_support_vectors = support_vectors[support_vectors_classes == 1]

        # находим весовые коэффициенты из выражения через двойственные коэффициенты
        # и пороговое значение через весовые коэффициенты и опорные вектора
        W = np.matmul((_lambda * A)[support_vectors_positions].T, support_vectors).reshape(2, 1)
        w_N = np.mean(support_vectors_classes - np.matmul(W.T, support_vectors.T))
        W = expand(W, w_N)

        # обучение модели SVC
        svc_clf = SVC(C=C, kernel='linear')
        svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

        # опорные вектора для метода SVC
        support_vectors_svc = svc_clf.support_vectors_
        support_vectors_svc_indices = svc_clf.support_
        support_vectors_svc_classes = dataset[support_vectors_svc_indices, -1, 0]
        red_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == -1]
        green_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == 1]

        # весовые коэффициенты и пороговое значение для модели SVC
        W_svc_clf = svc_clf.coef_.T
        w_N_svc_clf = svc_clf.intercept_[0]
        W_svc_clf = expand(W_svc_clf, w_N_svc_clf)

        print(f"C: {C}\n"
              f"W:\n{W}\n"
              f"W_svc_clf:\n{W_svc_clf}\n"
              f"Num of support vectors (solve_qp): {len(support_vectors)}\n"
              f"Num of support vectors (SVC): {len(support_vectors_svc)}")

        # строим разделяющую полосу и разделяющую гиперплоскость
        y = np.arange(-4, 4, 0.1)
        x = linear_border(y, W)
        x_svc_clf = linear_border(y, W_svc_clf)

        plot(f"solve_qp (cvxopt) C={C}", dataset3, dataset4, [x, x + 1 / W[0], x - 1 / W[0]], [y, y, y],
             ['black', 'green', 'red'], ['', '', ''])
        plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1], color='red')
        plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1], color='green')

        plot(f"SVC C={C}", dataset3, dataset4,
             [x_svc_clf, x_svc_clf + 1 / W_svc_clf[0], x_svc_clf - 1 / W_svc_clf[0]], [y, y, y],
             ['black', 'green', 'red'], ['', '', ''])
        plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1], color='red')
        plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1], color='green')
        plt.show()

# ===========================================

def get_discriminant_kernel(support_vectors, lambda_r, x, K, params):
    sum = 0
    for j in range(support_vectors.shape[0]):
        sum += lambda_r[j] * K(support_vectors[j].reshape(2, 1), x, params)
    return sum


def K_polynom(x, y, params):
    d = params[0]
    c = params[1]
    return pow(np.matmul(x.T, y)[0, 0] + c, d)


def get_P_kernel(dataset, N, K, params):
    P = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = r(dataset[j, :, :]) * r(dataset[i, :, :]) * K(dataset[j, :-1, :], dataset[i, :-1, :], params)
    return P

def task4(dataset3, dataset4):

    N = dataset3.shape[2] + dataset4.shape[2]

    # подготовка обучающей выборки
    dataset = get_train_dataset(dataset3, dataset4)

    # параметры для решения задачи квадратичного программирования
    kernel = 'poly'
    K = K_polynom
    params = [3, 1]

    P = get_P_kernel(dataset, N, K, params)
    q = np.full((N, 1), -1, dtype=np.double)
    G = np.concatenate((np.eye(N) * -1, np.eye(N)), axis=0)
    A = get_A(dataset, N)
    b = np.zeros(1)
    eps = 1e-04

    for C in [0.1, 1, 10]:
        h = np.concatenate((np.zeros((N,)), np.full((N,), C)), axis=0)

        # получаем вектор двойственных коэффициентов
        _lambda = solve_qp(P, q, G, h, A, b, solver='cvxopt')

        # опорные вектора для метода solve_qp
        support_vectors_positions = _lambda > eps
        support_vectors = dataset[support_vectors_positions, :-1, 0]
        support_vectors_classes = dataset[support_vectors_positions, -1, 0]
        red_support_vectors = support_vectors[support_vectors_classes == -1]
        green_support_vectors = support_vectors[support_vectors_classes == 1]

        # находим пороговое значение через ядро и опорные вектора
        w_N = []
        for j in range(support_vectors.shape[0]):
            w_N.append(get_discriminant_kernel(support_vectors, (_lambda * A)[support_vectors_positions],
                                             support_vectors[j].reshape(2, 1), K, params))
        w_N = np.mean(support_vectors_classes - np.array(w_N))

        # обучение модели SVC
        svc_clf = SVC(C=C, kernel=kernel, degree=3, coef0=1) # poly
        svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

        # опорные вектора для метода SVC
        support_vectors_svc = svc_clf.support_vectors_
        support_vectors_svc_indices = svc_clf.support_
        support_vectors_svc_classes = dataset[support_vectors_svc_indices, -1, 0]
        red_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == -1]
        green_support_vectors_svc = support_vectors_svc[support_vectors_svc_classes == 1]

        # строим разделяющую полосу и разделяющую гиперплоскость
        y = np.linspace(-4, 4, N)
        x = np.linspace(-4, 4, N)
        # создаем координатную сетку
        xx, yy = np.meshgrid(x, y)
        # получаем множество векторов
        xy = np.vstack((xx.ravel(), yy.ravel())).T

        # получаем множество значений решающей функции для нашей сетки (solve_qp)
        discriminant_func_values = []
        for i in range(xy.shape[0]):
            discriminant_func_values.append(get_discriminant_kernel(support_vectors,
                                                                  (_lambda * A)[support_vectors_positions],
                                                                  xy[i].reshape(2, 1), K, params)
                                            + w_N)
        discriminant_func_values = np.array(discriminant_func_values).reshape(xx.shape)

        # получаем множество значений решающей функции для нашей сетки (SVC)
        discriminant_func_values_svc = svc_clf.decision_function(xy).reshape(xx.shape)

        # # разделяющая полоса
        plot(f"solve_qp (cvxopt) ({kernel}) C={C}", dataset3, dataset4, [], [], ['black', 'green', 'red'], ['', '', ''])
        plt.contour(xx, yy, discriminant_func_values, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1], color='red')
        plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1], color='green')

        plot(f"SVC ({kernel}) C={C}", dataset3, dataset4, [], [], ['black', 'green', 'red'], ['', '', ''])
        plt.contour(xx, yy, discriminant_func_values_svc, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1], color='red')
        plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1], color='green')
        plt.show()


if __name__ == '__main__':

    X0 = np.load("resources/X0_0.npy")
    X1 = np.load("resources/X0_1.npy")

    # ===================
    #task2(X0, X1)

    X0 = np.load("resources/X0_1.npy")
    X1 = np.load("resources/X1_1.npy")

    #task3(X0, X1)
    task4(X0, X1)

    a = 3





