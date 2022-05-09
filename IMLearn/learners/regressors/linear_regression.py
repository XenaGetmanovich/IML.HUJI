from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        # checking if invertible
        if n_cols == n_rows and np.linalg.det(X) != 0:
            # invertible case
            X_T = np.transpose(X)
            self.coefs_ = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
        # not invertible case
        else:
            X_psudo_inv = pinv(X)
            self.coefs_ = X_psudo_inv.dot(y)

        if self.include_intercept_:
            # finding the intercept
            sum = 0
            for i in range(n_cols):
                sum += X[1][i]*self.coefs_[i]
            w_0 = y[1] - sum
            weights = [w_0]
            for i in range(n_cols):
                weights.append(self.coefs_[i])
            self.coefs_ = weights
        # raise NotImplementedError()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        n_samples = X.shape[0]
        n_feauters = X.shape[1]
        responses = np.zeros(n_samples)
        for i in range(n_samples):
            # creating a weights vector in length of number of feature + 1
            # to make the calculation simpler later on
            if self.include_intercept_:
                weights = self.coefs_
            else:
                weights = np.zeros(n_feauters)
                for j in range(n_feauters):
                    weights[j+1] = self.coefs_[j]
            # calculating the result for each sample
            sum = 0
            for j in range(n_feauters):
                sum += X[i][j] * weights[j+1]
            responses[i] = weights[0] + sum
        return responses
        # raise NotImplementedError()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """

        # fit the samples
        # predict the samples
        # calculate the sum of the losses
        self.fit(X, y)
        predictions = self.predict(X)

        def squared_loss(true_val, pre_val):
            return (true_val - pre_val) ^ 2

        sum = 0
        n_samples = y.size  # number of samples
        for i in range(n_samples):
            sum += squared_loss(y[i], predictions[i])
        return sum

        # raise NotImplementedError()
