import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

COLS = 1
DATA = 'date'
LONG = 'long'
LAT = 'lat'
PRICE = 'price'
ID = 'id'
DISTANCE_FROM_MEDIAN = 3
CATEGORICAL_ENCODING = "zipcode"


class LR:
    def __init__(self):
        self.data = None
        self.x = None
        self.w = None
        self.y = None
        self.training_data = None
        self.testing_data = None
        self.training_errors = []
        self.testing_errors = []

    def pre_process(self, path):
        self.data = pd.read_csv(path)
        self.clean_data()
        self.from_categorical_to_binary()

    def clean_data(self):
        self.data = self.data.drop([DATA, LAT, LONG], axis=COLS)
        self.data = self.data.dropna()
        self.data = self.data[(np.abs(stats.zscore(self.data)) < DISTANCE_FROM_MEDIAN).all(axis=COLS)]

    def from_categorical_to_binary(self):
        self.data = pd.get_dummies(self.data, prefix=CATEGORICAL_ENCODING, columns=[CATEGORICAL_ENCODING])

    def partition_to_train_test(self, percentage):
        self.training_data = self.data.sample(frac=percentage)
        self.testing_data = self.data.drop(self.training_data.index)

    def calculate_w(self):
        conjugate_transpose_x = np.linalg.pinv(self.x)
        self.w = conjugate_transpose_x.dot(self.y)

    def prepare_x_y(self, data):
        self.x = data.drop([PRICE, ID], axis=COLS)
        self.y = data[PRICE]

    def get_error(self):
        return (np.linalg.norm(self.x.dot(self.w) - self.y) ** 2) / np.shape(self.x)[0]

    def plot(self):
        points = np.arange(99)
        plt.scatter(points, np.log(self.training_errors))
        plt.scatter(points, np.log(self.testing_errors))
        plt.title("Training & Testing errors")
        plt.ylabel("log error")
        plt.xlabel("Test data percentage")
        plt.legend(('Training', 'Testing'))
        plt.show()


def get_singular_values(x, data):
    correlation_matrix = data.corr()
    xt = np.array(x.T)
    xxt = np.dot(x, xt)
    u, s, v = np.linalg.svd(xxt)
    print("u=", u, ", s=", s, ", v=", v)
    # print(correlation_matrix)


def main():
    path = 'kc_house_data.csv'
    lr = LR()
    lr.pre_process(path)

    for i in range(1, 100):
        lr.partition_to_train_test(float(i / 100))
        lr.prepare_x_y(lr.training_data)
        lr.calculate_w()
        lr.training_errors.append(lr.get_error())
        lr.prepare_x_y(lr.testing_data)
        lr.testing_errors.append(lr.get_error())

    lr.plot()


main()
