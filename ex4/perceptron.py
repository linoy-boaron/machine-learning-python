import numpy as np


class Perceptron:
    def __init__(self):
        self.final_w = None

    def fit(self, x, y):
        y_len = len(y)
        x_tag = np.insert(x, 2, 1, axis=1)
        w_tag = np.zeros(len(x_tag[0]))
        bool = True
        while bool:
            w = w_tag.copy()
            for i in range(y_len):
                c_r = y[i] * np.dot(w_tag, x_tag[i])
                if c_r <= 0:
                    w_tag = w_tag + y[i] * x_tag[i]
            bool = not np.array_equal(w, w_tag)
        self.final_w = w_tag
        return w_tag

    def predict(self, x):
        val = np.dot(x, self.final_w[:-1])
        if val < 0:
            return -1
        return 1

    def score(self, x, y):
        num = 0
        for i in range(len(x)):
            if self.predict(x[i]) == y[i]:
                num += 1
        return num / len(x)
