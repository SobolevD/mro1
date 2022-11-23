import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC

from utils.consts import EPS
from utils.sup_vectors import get_linear_classificator, get_train_dataset, get_P, get_A, concat, get_discr_kernel, \
    get_P_kernel, get_K, get_lambda, get_support_vectors, get_support_vector_classes


def draw_canvas(title, X0, X1, border_X, border_Y, colors, labels):

    plt.figure()
    plt.title(title)

    plt.plot(X0[0, :, :], X0[1, :, :], color='red',   marker='+')
    plt.plot(X1[0, :, :], X1[1, :, :], color='green', marker='*')

    for i in range(len(border_X)):
        plt.plot(border_X[i], border_Y[i], color=colors[i], label=labels[i])

    plt.legend()

    plt.xlim(left=-4,   right=4)
    plt.ylim(bottom=-4, top=4)


# Подсчет ошибки первого рода
def classification_error_type_1(dataset, W, class_num):

    errors = 0
    N = dataset.shape[-1]

    for i in range(N):
        z = concat(dataset[:, :, i], 1)
        if get_linear_classificator(z, W) != class_num:
            errors += 1

    return errors / N


def get_linear_border(y, W):
    a = W[0]
    b = W[1]
    c = W[2]

    if a == 0.:
        return 0
    else:
        return (-b * y - c) / a


def task2(dataset1, dataset2):

    N = dataset1.shape[2] + dataset2.shape[2]  # 100 + 100 = 200

    dataset = get_train_dataset(dataset1, dataset2)

    # Параметры для решения задачи квадратичного программирования
    # ==================== PARAMETERS ==================== #
    P = get_P(dataset, N)
    A = get_A(dataset, N)

    q = np.full((N, 1), -1, dtype=np.double)
    G = np.eye(N) * -1

    h = np.zeros((N,))
    b = np.zeros(1)
    # ==================== PARAMETERS ==================== #

    # ==================== [SOLVE QP] ==================== #
    # Вектор двойственных коэффициентов
    _lambda = get_lambda(P, q, G, h, A, b)

    # Опорные вектора вектора для метода solve_qp
    support_vectors_positions   = _lambda > EPS

    support_vectors             = get_support_vectors(dataset, support_vectors_positions)
    support_vectors_classes     = get_support_vector_classes(dataset, support_vectors_positions)

    red_support_vectors         = support_vectors[support_vectors_classes == -1]
    green_support_vectors       = support_vectors[support_vectors_classes == 1]

    # Весовые коэффициенты через двойственные коэффициенты
    W = np.matmul(
        (_lambda * A)[support_vectors_positions].T,
        support_vectors
    ).reshape(2, 1)

    # Пороговое значение через весовые коэффициенты и опорные вектора
    w_N = np.mean(support_vectors_classes - np.matmul(
        W.T, support_vectors.T))
    W = concat(W, w_N)
    # ==================== [SOLVE QP] ==================== #

    # ========================== [SVC] ========================== #
    svc_clf = SVC(C=(np.max(_lambda) + EPS), kernel='linear')
    svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    # опорные вектора для метода SVC
    support_vectors_svc             = svc_clf.support_vectors_

    support_vectors_svc_indices     = svc_clf.support_
    support_vectors_svc_classes     = dataset[support_vectors_svc_indices, -1, 0]

    red_support_vectors_svc         = support_vectors_svc[support_vectors_svc_classes == -1]
    green_support_vectors_svc       = support_vectors_svc[support_vectors_svc_classes == 1]

    # весовые коэффициенты и пороговое значение для модели SVC
    W_svc_clf   = svc_clf.coef_.T
    w_N_svc_clf = svc_clf.intercept_[0]
    W_svc_clf   = concat(W_svc_clf, w_N_svc_clf)
    # ========================== [SVC] ========================== #

    # ========================== [LinearSVC] ========================== #
    linear_svc_clf      = LinearSVC(C=(np.max(_lambda) + EPS))
    linear_svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    W_linear_svc_clf    = linear_svc_clf.coef_.T
    w_N_linear_svc_clf  = linear_svc_clf.intercept_[0]
    W_linear_svc_clf    = concat(W_linear_svc_clf, w_N_linear_svc_clf)
    # ========================== [LinearSVC] ========================== #

    print(f"W (Solve qp):\n{W}\n"
          f"W (SVC):\n{W_svc_clf}\n"
          f"W (Linear SVC):\n{W_linear_svc_clf}\n")

    # Разделяющая полосу и разделяющая гиперплоскость
    y           = np.arange(-4, 4, 0.1)
    x           = get_linear_border(y, W)
    x_svc       = get_linear_border(y, W_svc_clf)
    x_lin_svc   = get_linear_border(y, W_linear_svc_clf)

    draw_canvas(f"Solve QP", dataset1, dataset2, [x, x + 1 / W[0], x - 1 / W[0]], [y, y, y],
                ['black', 'green', 'red'], ['', '', ''])
    plt.scatter(red_support_vectors[:, 0],   red_support_vectors[:, 1],     color='red')
    plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1],   color='green')

    draw_canvas(f"SVC(kernel=linear)",
                dataset1, dataset2,
                [x_svc, x_svc + 1 / W_svc_clf[0],
                 x_svc - 1 / W_svc_clf[0]],
                [y, y, y],
                ['black', 'green', 'red'],
                ['', '', ''])

    plt.scatter(red_support_vectors_svc[:, 0],   red_support_vectors_svc[:, 1],     color='red')
    plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1],   color='green')

    draw_canvas(f"LinearSVC",
                dataset1, dataset2,
                [x_lin_svc, x_lin_svc + 1 / W_linear_svc_clf[0],
                 x_lin_svc - 1 / W_linear_svc_clf[0]],
                [y, y, y],
                ['black', 'green', 'red'],
                ['', '', ''])
    plt.show()


def task3(dataset3, dataset4):
    N = dataset3.shape[2] + dataset4.shape[2]  # 100 + 100 = 200

    # подготовка обучающей выборки
    dataset = get_train_dataset(dataset3, dataset4)

    # ==================== PARAMETERS ==================== #
    P = get_P(dataset, N)
    A = get_A(dataset, N)

    q = np.full((N, 1), -1, dtype=np.double)
    G = np.concatenate((np.eye(N) * -1, np.eye(N)), axis=0)
    b = np.zeros(1)
    # ==================== PARAMETERS ==================== #

    for C in [0.1, 1, 10, 20]:
        h = np.concatenate(
            (np.zeros((N,)), np.full((N,), C)),
            axis=0)

        # Вектор двойственных коэффициентов
        _lambda = get_lambda(P, q, G, h, A, b)

        support_vectors_positions   = _lambda > EPS

        support_vectors             = get_support_vectors(dataset, support_vectors_positions)
        support_vectors_classes     = get_support_vector_classes(dataset, support_vectors_positions)

        red_support_vectors         = support_vectors[support_vectors_classes == -1]
        green_support_vectors       = support_vectors[support_vectors_classes == 1]

        W = np.matmul(
            (_lambda * A)[support_vectors_positions].T,
            support_vectors).reshape(2, 1)
        w_N = np.mean(support_vectors_classes - np.matmul(W.T, support_vectors.T))
        W = concat(W, w_N)

        # ========================== [SVC] ========================== #
        svc_clf = SVC(C=C, kernel='linear')
        svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

        # опорные вектора для метода SVC
        support_vectors_svc         = svc_clf.support_vectors_

        support_vectors_svc_indices = svc_clf.support_
        support_vectors_svc_classes = dataset[support_vectors_svc_indices, -1, 0]

        red_support_vectors_svc     = support_vectors_svc[support_vectors_svc_classes == -1]
        green_support_vectors_svc   = support_vectors_svc[support_vectors_svc_classes == 1]

        W_svc   = svc_clf.coef_.T
        w_N_svc = svc_clf.intercept_[0]
        W_svc   = concat(W_svc, w_N_svc)
        # ========================== [SVC] ========================== #

        print(f"C: {C}\n"
              f"W (Solve qp):\n{W}\n"
              f"W (SVC):\n{W_svc}\n"
              f"Support vectors count (Solve qp): {len(support_vectors)}\n"
              f"Support vectors count (SVC): {len(support_vectors_svc)}")

        # Разделяющая полоса и разделяющая гиперплоскость
        y     = np.arange(-4, 4, 0.1)
        x     = get_linear_border(y, W)
        x_svc = get_linear_border(y, W_svc)

        draw_canvas(f"Solve QP (cvxopt) C={C}",
                    dataset3, dataset4,
                    [x, x + 1 / W[0], x - 1 / W[0]],
                    [y, y, y],
                    ['black', 'green', 'red'],
                    ['', '', ''])

        plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1],       color='red')
        plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1],   color='green')

        draw_canvas(f"SVC C={C}",
                    dataset3, dataset4,
                    [x_svc, x_svc + 1 / W_svc[0],
                     x_svc - 1 / W_svc[0]],
                    [y, y, y],
                    ['black', 'green', 'red'],
                    ['', '', ''])

        plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1],       color='red')
        plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1],   color='green')
        plt.show()

# ===========================================

def task4(dataset3, dataset4):

    N = dataset3.shape[2] + dataset4.shape[2]

    dataset = get_train_dataset(dataset3, dataset4)

    # ==================== PARAMETERS ==================== #
    kernel = 'poly'
    K = get_K
    params = [3, 1]

    P = get_P_kernel(dataset, N, params)
    A = get_A(dataset, N)

    q = np.full((N, 1), -1, dtype=np.double)
    G = np.concatenate((np.eye(N) * -1, np.eye(N)), axis=0)

    b = np.zeros(1)
    # ==================== PARAMETERS ==================== #

    for C in [0.1, 1, 10]:
        h = np.concatenate(
            (np.zeros((N,)), np.full((N,), C)),
            axis=0)

        # ==================== [Solve qp] ==================== #
        _lambda = get_lambda(P, q, G, h, A, b)

        # опорные вектора для метода solve_qp
        support_vectors_positions   = _lambda > EPS

        support_vectors             = get_support_vectors(dataset, support_vectors_positions)
        support_vectors_classes     = get_support_vector_classes(dataset, support_vectors_positions)

        red_support_vectors         = support_vectors[support_vectors_classes == -1]
        green_support_vectors       = support_vectors[support_vectors_classes == 1]

        # находим пороговое значение через ядро и опорные вектора
        w_N = []
        for j in range(support_vectors.shape[0]):
            w_N.append(get_discr_kernel(
                support_vectors,
                (_lambda * A)[support_vectors_positions],
                support_vectors[j].reshape(2, 1),
                params)
            )
        w_N = np.mean(support_vectors_classes - np.array(w_N))
        # ==================== [Solve qp] ==================== #


        # ==================== [SVC] ==================== #
        svc_clf = SVC(C=C, kernel=kernel, degree=3, coef0=1) # poly
        svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

        support_vectors_svc         = svc_clf.support_vectors_

        support_vectors_svc_pos = svc_clf.support_
        support_vectors_svc_classes = dataset[support_vectors_svc_pos, -1, 0]

        red_support_vectors_svc     = support_vectors_svc[support_vectors_svc_classes == -1]
        green_support_vectors_svc   = support_vectors_svc[support_vectors_svc_classes == 1]
        # ==================== [SVC] ==================== #

        # Разделяющая полоса и разделяющая гиперплоскость
        y = np.linspace(-4, 4, N)
        x = np.linspace(-4, 4, N)

        # Координатная сетка
        xx, yy = np.meshgrid(x, y)

        # Множество векторов
        xy = np.vstack((xx.ravel(), yy.ravel())).T

        # Множество значений решающей функции для сетки (solve_qp)
        discriminant_func_values = []
        for i in range(xy.shape[0]):
            discriminant_func_values.append(
                get_discr_kernel(
                    support_vectors,
                    (_lambda * A)[support_vectors_positions],
                    xy[i].reshape(2, 1), params
                ) + w_N)

        discriminant_func_values = np.array(discriminant_func_values).reshape(xx.shape)

        # Множество значений решающей функции для сетки (SVC)
        discriminant_func_values_svc = svc_clf.decision_function(xy).reshape(xx.shape)

        # Разделяющая полоса
        draw_canvas(f"Solve QP ({kernel}) C={C}",
                    dataset3, dataset4,
                    [], [],
                    ['black', 'green', 'red'],
                    ['', '', ''])

        plt.contour(xx, yy, discriminant_func_values, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        plt.scatter(red_support_vectors[:, 0], red_support_vectors[:, 1], color='red')
        plt.scatter(green_support_vectors[:, 0], green_support_vectors[:, 1], color='green')

        draw_canvas(f"SVC ({kernel}) C={C}",
                    dataset3, dataset4,
                    [], [],
                    ['black', 'green', 'red'],
                    ['', '', ''])

        plt.contour(xx, yy, discriminant_func_values_svc, levels=[-1, 0, 1], colors=['red', 'black', 'green'])
        plt.scatter(red_support_vectors_svc[:, 0], red_support_vectors_svc[:, 1], color='red')
        plt.scatter(green_support_vectors_svc[:, 0], green_support_vectors_svc[:, 1], color='green')
        plt.show()


if __name__ == '__main__':

    X0_0 = np.load("resources/X0_0.npy")
    X0_1 = np.load("resources/X0_1.npy")
    X1_0 = np.load("resources/X0_1.npy")
    X1_1 = np.load("resources/X1_1.npy")

    task2(X0_0, X0_1)
    task3(X1_0, X1_1)
    task4(X1_0, X1_1)






