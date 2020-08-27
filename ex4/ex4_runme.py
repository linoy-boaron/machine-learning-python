import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from Perceptron import Perceptron
import ex4_tools
from adaboost import AdaBoost


class ex5:
    def __init__(self):
        self.mean = [0, 0]
        self.cov = np.eye(2)
        self.svm = SVC(C=1e10, kernel='linear')
        self.perceptron = None
        self.a_boost = None
        self.svm_accs = []
        self.perceptrone_accs = []
        self.ms = [5, 10, 15, 25, 70]
        self.ts = [5, 10, 50, 100, 200, 500]

    def q_3_4_5(self):
        for m in self.ms:
            self.calculate_for_m(m)
        plt.plot(self.ms, self.perceptrone_accs)
        plt.plot(self.ms, self.svm_accs)
        plt.legend(("perceptron","svd"))
        plt.show()

    def calculate_for_m(self, m):
        x = np.random.multivariate_normal(self.mean, self.cov, m)
        real_labels = self.get_real_labels(x)
        labeled_1_x, labeled_min_1_x = self.get_x_by_labels(x, real_labels)
        t = np.arange(int(x.min()) - 1, int(x.max()) + 1, 0.1)
        self.plt_xs(labeled_1_x, labeled_min_1_x, t)
        self.perceptron = Perceptron()
        perc_w = self.perceptron.fit(x, real_labels)
        plt.plot(t, self.get_y(perc_w[:-1], perc_w[-1], t))
        self.svm.fit(x, real_labels)
        plt.plot(t, self.get_y(self.svm._get_coef()[0], self.svm.intercept_, t))
        plt.legend(["True labels", "perceptron", "svm"])
        plt.show()
        self.calculate_svm_perc_acc()

    def get_real_labels(self, x):
        labels = []
        for j in x:
            labels.append(self.f(j))
        return labels

    def get_x_by_labels(self, x, labels):
        x_1, x_minus_1 = [], []
        for i in range(len(x)):
            if labels[i] == 1.0:
                x_1.append(x[i])
            elif labels[i] == -1.0:
                x_minus_1.append(x[i])
            else:
                pass
        return x_1, x_minus_1

    def f(self, x):
        return np.sign(np.dot([0.3, -0.5], x) + 0.1)

    def plt_xs(self, labeled_1_x, labeled_min_1_x, t):
        plt.scatter([x[0] for x in labeled_1_x], [x[1] for x in labeled_1_x])
        plt.scatter([x[0] for x in labeled_min_1_x], [x[1] for x in labeled_min_1_x])
        plt.plot(t, self.get_y([0.3, -0.5], 0.1, t))

    def get_y(self, w, b, x):
        y = []
        for i in x:
            y.append(-w[0] * i / w[1] + b / -w[1])
        return y

    def calculate_svm_perc_acc(self):
        s,p = self.get_svm_prec_acc()
        self.perceptrone_accs.append(p / 500)
        self.svm_accs.append(s / 500)

    def get_svm_prec_acc(self):
        svm_acc, perceptrone_acc = 0, 0
        for i in range(500):
            x = np.random.multivariate_normal(self.mean, self.cov, 10000)
            real_labels = []
            for j in x:
                real_labels.append(self.f(j))
            svm_acc += self.svm.score(x, real_labels)
            perceptrone_acc += self.perceptron.score(x, real_labels)
        return svm_acc, perceptrone_acc

    def q_7_8_9_10(self):
        self.q_8()
        self.q_9()
        self.q_10()

    def q_8(self):
        tx,ty = ex4_tools.generate_data(5000, noise_ratio=0)
        x, y = ex4_tools.generate_data(200, noise_ratio=0)
        self.a_boost = AdaBoost(WL=ex4_tools.DecisionStump, T=500)
        self.a_boost.train(tx, ty)
        training_errs, test_errs = self.get_ab_errs(tx,ty, x, y)
        self.plt_q_8(training_errs, test_errs)

    def get_ab_errs(self, tx,ty, x, y):
        training_errs, test_errs = [], []
        for i in range(500):
            training_errs.append(self.a_boost.error(tx, ty, i))
            test_errs.append(self.a_boost.error(x, y, i))
        return training_errs, test_errs

    def plt_q_8(self, training_errs, test_errs):
        plt.plot(np.arange(500), training_errs, label="training error")
        plt.plot(np.arange(500), test_errs, label="test error")
        plt.title("Adaboost errors as function of (T)")
        plt.legend()
        plt.show()

    def q_9(self):
        tx, ty = ex4_tools.generate_data(5000, noise_ratio=0)
        x, y = ex4_tools.generate_data(200, noise_ratio=0)
        i = 1
        for t in self.ts:
            a_boost = AdaBoost(WL=ex4_tools.DecisionStump, T=t)
            a_boost.train(tx,ty)
            plt.subplot(2, 3, i)
            ex4_tools.decision_boundaries(a_boost, x, y, t)
            i += 1
        plt.show()

    def q_10(self):
        tx, ty = ex4_tools.generate_data(5000, noise_ratio=0)
        x, y = ex4_tools.generate_data(200, noise_ratio=0)
        errors = self.get_ab_errors(tx,ty, x, y)
        min_t = np.argmin(errors)
        a_boost = AdaBoost(WL=ex4_tools.DecisionStump, T=self.ts[min_t])
        a_boost.train(tx,ty)
        ex4_tools.decision_boundaries(a_boost, tx,ty, self.ts[min_t])
        plt.title("min error is " + str(errors[min_t]) + " with " + str(self.ts[min_t]) + " classifiers")
        plt.show()

    def get_ab_errors(self, tx,ty, x, y):
        errors = []
        for t in self.ts:
            a_boost = AdaBoost(WL=ex4_tools.DecisionStump, T=t)
            a_boost.train(tx,ty)
            errors.append(a_boost.error(x, y, t))
        return errors



def main():
    a = ex5()
    a.q_3_4_5()
    a.q_7_8_9_10()



