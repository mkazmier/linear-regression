**ENG**
A Python-numpy implementation of linear regression using ordinary least squares fit.
Requires Python 3.0 or higher and numpy 1.10 or higher.
Run the script without arguments to see available options.
The algorithm is contained in the OrdinaryLeastSquares class, so it can be imported and used in custom programs like so:

```python
    from script_file_name import OrdinaryLeastSquares
    reg = OrdinaryLeastSquares()
```

The algorithm uses scikit-learn style fit-predict API:

```python
    # X_train, X_test should be numpy arrays with shape (n_examples, n_features)
    # y_train should be a numpy array with shape (n_examples, 1)
    reg.fit(X_train, y_train)
    reg.predict(X_test)
```

**Resources consulted:**:
Prof. Yaser Abu-Moustafa's  lecture on linear models from Caltech's introductory machine learning course:
[https://www.youtube.com/watch?v=FIbVs5GbBlQ]

Wikipedia article on least squares fit:
[https://en.wikipedia.org/wiki/Least_squares]

Wolfram MathWorld entry on the subject:
[http://mathworld.wolfram.com/LeastSquaresFitting.html]

scikit-learn and numpy documentation:
[http://scikit-learn.org/stable/documentation.html]
[http://docs.scipy.org/doc/numpy/index.html]
