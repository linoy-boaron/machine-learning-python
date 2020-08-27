import numpy as np

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.T = T
        self.WL = WL
        self.h = [None for i in range(T)]
        self.w = np.zeros(T)

    def new_D(self, D, p, w_t, y):
        return D * np.exp(-np.multiply(y, p) * w_t) / np.sum \
            (D * np.exp(np.multiply(y, p) * w_t))

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        size = y.shape[0]
        D = np.full(size, 1 / size)
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            self.w[t] = 0.5 * np.log((1 / (np.sum(D * np.not_equal(self.h[t].predict(X), y)))) - 1)
            D = self.new_D(D,self.h[t].predict(X),self.w[t],y)


    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predictionValues = np.zeros(X.shape[0])
        for i in range(max_t):
            prediction = self.h[i].predict(X)
            predictionValues += prediction*self.w[i]
        h_predict = np.sign(predictionValues)
        return h_predict

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """

        predictions = (self.predict(X, max_t) != y).sum()
        size = y.shape[0]
        return predictions / size
