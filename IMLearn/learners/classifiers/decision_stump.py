from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.sign_ = 1
        self.threshold_ = (self._find_threshold(X[:, self.j_], y, self.sign_))[0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # go over each value, check if it's above or below the threshold
        n_samples = X.shape[0]
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if X[i][self.j_] < self.threshold_:
                y[i] = -self.sign_
            else:
                y[i] = self.sign_
        return y

        # raise NotImplementedError()

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # for each element in the feature column we want to predict labels, calculate loss, compare
        # to the best one
        n_samples = values.shape[0]
        best_thresh = 0
        best_thresh_mistakes = n_samples + 1  # setting the number of mistakes to be more than
        # possible so that the best threshold will be replaced
        best_sign = 1

        # predicts the labels given a threshold and a column of values
        def local_predict(value_column, threshold):
            n_samples = value_column.shape[0]  # number of samples
            y = np.zeros(n_samples)
            for j in range(n_samples):
                if value_column[j] < threshold:
                    y[j] = -1 * self.sign_
                else:
                    y[j] = self.sign_
            return y

        # calculate the misclacification loss given predicted, and real labels
        def local_loss(y_true, y_pred):
            n_samples = y_true.shape[0]
            num_of_mistakes = 0
            for i in range(n_samples):
                if y_pred[i] != y_true[i]:
                    num_of_mistakes += 1
            return num_of_mistakes
            # DELETE
        sorted_y = labels
        sorted_y.sort()
        for i in range(n_samples):
            curr_thresh = values[i]
            curr_prediction = local_predict(values, curr_thresh)
            curr_thresh_mistakes = local_loss(labels, curr_prediction)
            # compare to best
            if curr_thresh_mistakes < best_thresh_mistakes:
                best_thresh = curr_thresh
                best_thresh_mistakes = curr_thresh_mistakes
                best_sign = self.sign_
                #if switching the signs yields a better result we choose this sign
            if n_samples - curr_thresh_mistakes < best_thresh_mistakes:
                best_thresh = curr_thresh
                best_thresh_mistakes = n_samples - curr_thresh_mistakes
                best_sign = -1 * self.sign_
        self.sign_ = best_sign
        return best_thresh, best_thresh_mistakes

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        num_of_mistakes = 0
        for i in range(n_samples):
            if y_pred[i] != y[i]:
                num_of_mistakes += 1
        return num_of_mistakes
