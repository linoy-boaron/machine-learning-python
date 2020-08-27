import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
import math


def q_5_1(mean):
    x = np.linspace(mean - 2, mean + 2, 100)
    plt.plot(x, stats.norm.pdf(x, mean, 1))
    plt.plot(x, stats.norm.cdf(x, mean, 1))
    plt.show()


def q_5_2_3_4(m_1, m_2, i):
    x_1, x_2, y_1, y_2 = None, None, None, None
    if i == 2:
        x_1, y_1 = q_5_2(m_1)
        x_2, y_2 = q_5_2(m_2)
    elif i == 3:
        x_1, y_1 = q_5_3(m_1)
        x_2, y_2 = q_5_3(m_2)
    elif i == 4:
        x_1, y_1 = q_5_4(m_1)
        x_2, y_2 = q_5_4(m_2)
    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)
    plt.show()


def q_5_2(mean):
    x = np.linspace(0.000001, 0.9999999999999, 100)
    y = [(math.log2(i / (1 - i)) + 10) / 2 for i in x]
    return x, stats.norm.cdf(y, mean, 1)


def q_5_3(mean):
    x = np.linspace(0.0001, 0.99999, 1000)
    y = [5 + 0.5 * math.log2(i / (1 - i)) for i in x]
    return x, stats.norm.cdf(y, mean, 1)


def q_5_4(mean):
    x = np.linspace(0.0001, 0.99999, 1000)
    y = [5 + 0.5 * math.log2(i / (1 - i)) for i in x]
    h_of_x = [(math.log2(i / (1 - i)) + 10) / 2 for i in x]
    return h_of_x, stats.norm.cdf(y, mean, 1)


def question_5_5(mean_1, mean_2):
    vals = [0.2, 0.4, 0.55, 0.95]
    for x in vals:
        y = (5 + math.log2(x / (1 - x)) / 2)
        fpr = 1 - stats.norm.cdf(y, mean_1, 1)
        tpr = 1 - stats.norm.cdf(y, mean_2, 1)

        print("fpr at " + str(x) + " is: " + str(fpr))
        print("tpr at " + str(x) + " is: " + str(tpr))


def question_5_6(mean_1, mean_2):
    vals = [0.2, 0.4, 0.55, 0.95]
    for x in vals:
        y = (5 + math.log2(x / (1 - x)) / 2)
        plt.axvline(x=y)
    plot_x_i(mean_1, 'tab:red')
    plot_x_i(mean_2, 'tab:blue')
    plt.legend(('t=0.2', 't=0.4', 't=0.55', 't=0.95', 'mean=' + str(mean_1), 'mean=' + str(mean_2)))
    plt.show()


def plot_x_i(mean, color):
    x_i = np.linspace(mean - 3, mean + 3, 100)
    plt.plot(x_i, stats.norm.pdf(x_i, mean, 1), color)


def question_5_7(mean_1, mean_2):
    x = np.linspace(0.0001, 0.99999, 1000)
    y = []
    for i in x:
        y.append((5 + (math.log2(i / (1 - i))) / 2))
        plt.plot(1 - stats.norm.cdf(y, mean_1, 1), 1 - stats.norm.cdf(y, mean_2, 1))
    plt.show()


def q_7_a_b():
    for x in range(10):
        train_set_x, train_set_y, test_set_x, test_set_y = get_and_process_data("spam.data.txt")
        lr = LogisticRegression(solver="liblinear").fit(train_set_x, train_set_y.reshape(len(train_set_x)))
        pb = (lr.predict_proba(test_set_x))[:, 0]
        sorted_indexes = np.argsort(pb)
        test_set_y = test_set_y[sorted_indexes]

        n_p, n_n, n_i = get_np_nn_ni(test_set_y)
        plot_tpr_fpr(n_p, n_i, n_n)

    plt.title("ROC for 10 examples")
    plt.ylabel("tpr")
    plt.xlabel("fpr")
    plt.show()


def get_and_process_data(path):
    data = np.genfromtxt(path, delimiter=" ")
    index_test = random.sample(range(len(data)), 1000)
    test_set = data[index_test]
    train_set = np.delete(data, index_test, 0)
    return train_set[:, :-1], train_set[:, -1:], test_set[:, :-1], test_set[:, -1:]


def get_np_nn_ni(test_set_y):
    test_set_y_len = len(test_set_y)
    n_p = np.sum(test_set_y)
    n_i = []
    for i in range(1, int(n_p)):
        n_i.append(get_i_n_i(i, test_set_y, test_set_y_len))
    return int(n_p), test_set_y_len - n_p, n_i


def get_i_n_i(i, test_set_y, test_set_y_len):
    k = 0
    for j in range(test_set_y_len):
        if int(test_set_y[j]) == 1:
            k += 1
            if k == i:
                return j


def plot_tpr_fpr(n_p, n_i, n_n):
    tpr = [(i / n_p) for i in range(n_p - 1)]
    fpr = [(n_i[i] - i) / n_n for i in range(n_p - 1)]
    plt.plot(fpr, tpr)


def main():
    mean_1 = 6
    mean_2 = 4
    q_5_1(mean_1)
    q_5_1(mean_2)
    for i in range(2, 5):
        q_5_2_3_4(mean_1, mean_2, i)
    question_5_5(mean_1, mean_2)
    question_5_6(mean_1, mean_2)
    question_5_7(mean_1, mean_2)
    q_7_a_b()


