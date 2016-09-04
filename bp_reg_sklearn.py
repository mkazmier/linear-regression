#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from sys import argv, exit

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
        reg = LinearRegression()

        # load training examples
        with open(X_train_file, 'r') as f:
            X_train = np.array([float(x) for x in f]).reshape(-1, 1)

        with open(y_train_file, 'r') as f:
            y_train = np.array([float(y) for y in f]).reshape(-1, 1)

        # fit the model on training data and make predictions on test data
        reg.fit(X_train, y_train)
        predicted = reg.predict(X_test)
        print("Predicted output:\n{}".format(predicted.reshape(-1,1)))

    # if test outputs are given for evaluation, calculate the r^2 score of regression
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
