import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC

from utils.baios import get_bayos_lines
from utils.consts import EPS, B0, B1, M0, M1
from utils.sup_vectors import get_linear_classificator, get_train_dataset, get_P, get_A, concat, get_discr_kernel, \
    get_P_kernel, get_lambda, get_support_vectors, get_support_vector_classes


def draw_canvas(title, X0, X1, border_X, border_Y, colors):

    plt.figure()
    plt.title(title)

    plt.plot(X0[0, :, :], X0[1, :, :], color='red',  marker='.')
    plt.plot(X1[0, :, :], X1[1, :, :], color='blue', marker='.')

    for i in range(len(border_X)):
        plt.plot(border_X[i], border_Y[i], color=colors[i])

    plt.legend()

    plt.xlim(left=-4,   right=4)
    plt.ylim(bottom=-4, top=4)


def draw_canvas_3(title, X0, X1,
                  border_X_1, border_Y_1,
                  border_X_2, border_Y_2,
                  border_X_3, border_Y_3,
                  colors_1, colors_2, colors_3):

    plt.figure()
    plt.title(title)

    plt.plot(X0[0, :, :], X0[1, :, :], color='red',  marker='.')
    plt.plot(X1[0, :, :], X1[1, :, :], color='blue', marker='.')

    for i in range(len(border_X_1)):
        plt.plot(border_X_1[i], border_Y_1[i], color=colors_1[i], ds='steps')
        plt.plot(border_X_2[i], border_Y_2[i], color=colors_2[i])
        plt.plot(border_X_3[i], border_Y_3[i], color=colors_3[i])

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

    support_vectors_class_1     = support_vectors[support_vectors_classes == -1]
    support_vectors_class_2     = support_vectors[support_vectors_classes == 1]

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

    support_vectors_svc_class_1     = support_vectors_svc[support_vectors_svc_classes == -1]
    support_vectors_svc_class_2     = support_vectors_svc[support_vectors_svc_classes == 1]

    # весовые коэффициенты и пороговое значение для модели SVC
    W_svc   = svc_clf.coef_.T
    w_N_svc = svc_clf.intercept_[0]
    W_svc   = concat(W_svc, w_N_svc)
    # ========================== [SVC] ========================== #

    # ========================== [LinearSVC] ========================== #
    linear_svc      = LinearSVC(C=(np.max(_lambda) + EPS))
    linear_svc.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

    W_linear_svc    = linear_svc.coef_.T
    w_N_linear_svc  = linear_svc.intercept_[0]
    W_linear_svc    = concat(W_linear_svc, w_N_linear_svc)
    # ========================== [LinearSVC] ========================== #

    print(f"W (Solve qp):\n{W}\n"
          f"W (SVC):\n{W_svc}\n"
          f"W (Linear SVC):\n{W_linear_svc}\n")

    # Разделяющая полосу и разделяющая гиперплоскость
    y           = np.arange(-4, 4, 0.1)
    x           = get_linear_border(y, W)
    x_svc       = get_linear_border(y, W_svc)
    x_lin_svc   = get_linear_border(y, W_linear_svc)

    draw_canvas_3(f"Solve QP & SVC & SVC Linear",
                dataset1, dataset2,

                [x, x + 1 / W[0], x - 1 / W[0]],
                [y, y, y],

                  [x_svc, x_svc + 1 / W_svc[0],
                   x_svc - 1 / W_svc[0]],
                   [y, y, y],

                  [x_lin_svc, x_lin_svc + 1 / W_linear_svc[0],
                   x_lin_svc - 1 / W_linear_svc[0]],
                  [y, y, y],

                    ['black', 'blue', 'red'],
                  ['green', 'yellow', 'orange'],
                  ['brown', 'purple', 'gray'])

    plt.scatter(support_vectors_class_1[:, 0], support_vectors_class_1[:, 1], color='red')
    plt.scatter(support_vectors_class_2[:, 0], support_vectors_class_2[:, 1], color='blue')

    plt.scatter(support_vectors_svc_class_1[:, 0], support_vectors_svc_class_1[:, 1],   color='red')
    plt.scatter(support_vectors_svc_class_2[:, 0], support_vectors_svc_class_2[:, 1],   color='blue')


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

    for C in [0.1, 1, 10, 35]:
        h = np.concatenate(
            (np.zeros((N,)), np.full((N,), C)),
            axis=0)

        # Вектор двойственных коэффициентов
        _lambda                     = get_lambda(P, q, G, h, A, b)

        support_vectors_positions   = _lambda > EPS

        support_vectors             = get_support_vectors(dataset, support_vectors_positions)
        support_vectors_classes     = get_support_vector_classes(dataset, support_vectors_positions)

        support_vectors_class_1     = support_vectors[support_vectors_classes == -1]
        support_vectors_class_2     = support_vectors[support_vectors_classes == 1]

        W   = np.matmul(
            (_lambda * A)[support_vectors_positions].T,
            support_vectors).reshape(2, 1)

        w_N = np.mean(support_vectors_classes - np.matmul(W.T, support_vectors.T))
        W   = concat(W, w_N)

        # ========================== [SVC] ========================== #
        svc_clf                         = SVC(C=C, kernel='linear')
        svc_clf.fit(dataset[:, :-1, 0], dataset[:, -1, 0])

        # опорные вектора для метода SVC
        support_vectors_svc             = svc_clf.support_vectors_

        support_vectors_svc_indices     = svc_clf.support_
        support_vectors_svc_classes     = dataset[support_vectors_svc_indices, -1, 0]

        support_vectors_svc_class_1     = support_vectors_svc[support_vectors_svc_classes == -1]
        support_vectors_svc_class_2     = support_vectors_svc[support_vectors_svc_classes == 1]

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

        draw_canvas(f"Solve QP C={C}",
                    dataset3, dataset4,
                    [x, x + 1 / W[0], x - 1 / W[0]],
                    [y, y, y],
                    ['black', 'blue', 'red'])

        plt.scatter(support_vectors_class_1[:, 0], support_vectors_class_1[:, 1],       color='red')
        plt.scatter(support_vectors_class_2[:, 0], support_vectors_class_2[:, 1],   color='blue')

        draw_canvas(f"SVC C={C}",
                    dataset3, dataset4,
                    [x_svc, x_svc + 1 / W_svc[0],
                     x_svc - 1 / W_svc[0]],
                    [y, y, y],
                    ['black', 'blue', 'red'])

        plt.scatter(support_vectors_svc_class_1[:, 0], support_vectors_svc_class_1[:, 1],       color='red')
        plt.scatter(support_vectors_svc_class_2[:, 0], support_vectors_svc_class_2[:, 1],   color='blue')
        plt.show()

# ===========================================

def task4(dataset3, dataset4):

    N = dataset3.shape[2] + dataset4.shape[2]

    dataset = get_train_dataset(dataset3, dataset4)

    # ==================== PARAMETERS ==================== #
    kernel = 'poly'
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

        support_vectors_class_1     = support_vectors[support_vectors_classes == -1]
        support_vectors_class_2     = support_vectors[support_vectors_classes == 1]

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

        support_vectors_svc             = svc_clf.support_vectors_

        support_vectors_svc_pos         = svc_clf.support_
        support_vectors_svc_classes     = dataset[support_vectors_svc_pos, -1, 0]

        support_vectors_svc_class_1     = support_vectors_svc[support_vectors_svc_classes == -1]
        support_vectors_svc_class_2     = support_vectors_svc[support_vectors_svc_classes == 1]
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
                    ['black', 'blue', 'red'])

        plt.contour(xx, yy, discriminant_func_values, levels=[-1, 0, 1], colors=['red', 'black', 'blue'])
        plt.scatter(support_vectors_class_1[:, 0], support_vectors_class_1[:, 1], color='red')
        plt.scatter(support_vectors_class_2[:, 0], support_vectors_class_2[:, 1], color='blue')

        baios_x = np.linspace(-4, 4, 200).reshape(200, 1)
        baios_y = np.zeros(200).reshape(200, 1)
        for t in range(0, len(baios_x)):
            _, baios_y[t] = get_bayos_lines(B0, B1, M0, M1, 0.5, 0.5, baios_x[t])

        plt.scatter(baios_x, baios_y, color='purple')

        draw_canvas(f"SVC ({kernel}) C={C}",
                    dataset3, dataset4,
                    [], [],
                    ['black', 'blue', 'red'])

        plt.contour(xx, yy, discriminant_func_values_svc, levels=[-1, 0, 1], colors=['red', 'black', 'blue'])
        plt.scatter(support_vectors_svc_class_1[:, 0], support_vectors_svc_class_1[:, 1], color='red')
        plt.scatter(support_vectors_svc_class_2[:, 0], support_vectors_svc_class_2[:, 1], color='blue')

        baios_x = np.linspace(-4, 4, 200).reshape(200, 1)
        baios_y = np.zeros(200).reshape(200, 1)
        for t in range(0, len(baios_x)):
            _, baios_y[t] = get_bayos_lines(B0, B1, M0, M1, 0.5, 0.5, baios_x[t])

        plt.scatter(baios_x, baios_y, color='purple')

        plt.show()

        # вероятности ошибочной классификации
        p0 = 0.
        p1 = 0.
        for i in range(dataset3.shape[2]):
            if get_discr_kernel(support_vectors, (_lambda * A)[support_vectors_positions], dataset3[:, :, i], params) + w_N > 0:
                p0 += 1
            if get_discr_kernel(support_vectors, (_lambda * A)[support_vectors_positions], dataset4[:, :, i], params) + w_N < 0:
                p1 += 1
        p0 /= dataset3.shape[2]
        p1 /= dataset3.shape[2]

        print(f"solve_qp(osqp) C={C} kernel({kernel}) p0: {p0} p1: {p1}")

        # вероятности ошибочной классификации
        test_dataset3 = dataset[dataset[:, -1, 0] < 0, :-1, 0]
        test_dataset4 = dataset[dataset[:, -1, 0] > 0, :-1, 0]

        predicted_class3 = svc_clf.predict(test_dataset3)
        predicted_class4 = svc_clf.predict(test_dataset4)

        p0 = np.sum(predicted_class3 > 0) / test_dataset3.shape[0]
        p1 = np.sum(predicted_class4 < 0) / test_dataset4.shape[0]

        print(f"SVC C={C} kernel({kernel}) p0: {p0} p1: {p1}")


if __name__ == '__main__':

    X0_0 = np.load("resources/X0_0.npy")
    X0_1 = np.load("resources/X0_1.npy")
    X1_0 = np.load("resources/X0_1.npy")
    X1_1 = np.load("resources/X1_1.npy")

    #task2(X0_0, X0_1)
    #task3(X1_0, X1_1)
    task4(X1_0, X1_1)






