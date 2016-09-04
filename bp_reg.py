#!/usr/bin/env python3

import numpy as np
import pickle
from sys import argv, exit

class OrdinaryLeastSquares(object):

    """Computes linear regresion using ordinary least squares method.
       The parameters of regression are computed using:

                w = X_dagger * y

        where X_dagger is the Moore-Penrose pseudoinverse of the input matrix:

                X_dagger = ((X.T * X) ** -1) * X.T)

        Implementation based on lecture by Prof. Yaser Abu-Mostafa
        during Caltech introductory machine learning course, available at
        https://www.youtube.com/watch?v=FIbVs5GbBlQ"""

    def _least_squares(self, X, y):

        """Computes the input matrix pseudoinverse
           and returns the model parameters"""
        
        # add the x_0 coordinate representing the bias (intercept).
        # This is done to simplify the calculations.
        X = add_ones(X)
        
        # Note: the straightforward implementation of pseudoinverse
        # formula produces inacurate results due to precision errors.
        # np.linalg.pinv approximates the pseudoinverse using SVD.
        # Read more:
        # https://bytes.com/topic/python/answers/627919-error-matrix-inversion-using-numpy
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html
        X_dagger = np.linalg.pinv(X)
        return np.dot(X_dagger, y)

    def fit(self, X_train, y_train):
        """Fit the regression to the training data."""
        self.w = self._least_squares(X_train, y_train)

    def predict(self, X_test):
        """Predict the output for X_test. Train the model using 
           self.fit on training data before predicting."""
        X_test = add_ones(X_test)
        return np.dot(self.w.T, X_test.T).T

    def score(self, X_test, y_test):
        """Compute the R^2 score of the regression."""
        y_pred = self.predict(X_test)
        ss_res = np.linalg.norm(y_test - y_pred) ** 2
        ss_tot = np.linalg.norm(y_test - y_test.mean()) ** 2
        return 1 - (ss_res / ss_tot)
    

# misc functions
def add_ones(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def main():
    try:
        if '-l' not in argv:
            X_train_file = argv[1]
            y_train_file = argv[2]
        X_test_file = argv[3]
        options = {'-t': None, '-l': None, '-s': None}
        for o in options.keys():
            if o in argv:
                options[o] = argv[argv.index(o)+1]
        
    except IndexError:
        print("usage: {} X_train_file y_train_file | -l model_file X_test_file [-t y_test_file] [-s save_file]".format(argv[0]))
        exit(0)           

    # load test examples
    with open(X_test_file, 'r') as f:
            X_test = np.array([float(x) for x in f]).reshape(-1, 1)

    # load pickled model from file
    if options['-l']:
        with open(options['-l'], 'rb') as f:
            reg = pickle.load(f)
            print("Loaded model form {}".format(options['-l']))
            predicted = reg.predict(X_test)
            print("Predicted output:\n{}".format(predicted.reshape(-1,1)))
    else:
        reg = OrdinaryLeastSquares()

        # load training examples
        with open(X_train_file, 'r') as f:
            X_train = np.array([float(x) for x in f]).reshape(-1, 1)

        with open(y_train_file, 'r') as f:
            y_train = np.array([float(y) for y in f]).reshape(-1, 1)

        # fit the model on training data and make predictions on test data
        reg.fit(X_train, y_train)
        predicted = reg.predict(X_test)
        print("Predicted output:\n{}".format(predicted.reshape(-1,1)))

    # if test outputs are given for evaluation, calculate the R^2 score of regression
    if options['-t']:
        with open(options['-t'], 'r') as f:
            y_test = np.array([float(y) for y in f]).reshape(-1, 1)

        score = reg.score(X_test, y_test)
        print("R^2 score: {}".format(score))

    if options['-s']:
        with open(options['-s'], 'wb+') as f:
            pickle.dump(reg, f)
        print("Saved model to {}".format(options['-s']))


if __name__ == '__main__':
    main()
