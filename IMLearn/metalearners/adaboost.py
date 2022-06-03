import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # calculates the weight of an estimator

        def resample(curr_data: np.ndarray, curr_labels: np.ndarray):
            n_samples =  curr_data.shape[0]
            n_features = curr_data.shape[1]
            index_vector = np.arange(n_samples)
            resampled_index_vector = np.random.choice(index_vector, n_samples,True,self.D_)
            resampled_index_vector.sort()
            new_data = np.zeros((n_samples, n_features))
            new_labels = np.zeros(n_samples)
            for i in range(n_samples):
                new_data[i] = curr_data[resampled_index_vector[i]]
                new_labels[i] = curr_labels[resampled_index_vector[i]]
            return (new_data, new_labels)

        def find_weight(pred_labels: np.ndarray, y_true: np.ndarray, index: int):
            n_samples = curr_data.shape[0]
            curr_estimator = self.models_[index]
            # weighted number of mistakes
            w_n_mistakes = 0
            for i in range(n_samples):
                if pred_labels[i] != y_true[i]:
                    w_n_mistakes += self.D_[i]
            if w_n_mistakes == 0:
                return 0
            a = 0.5*(np.log(np.abs(1 - (1/w_n_mistakes))))
            return a
            # return 0.5*(np.log((1/w_n_mistakes) - 1))

        # finds the decision stump that returns the lowest number of mistakes on the curr_data
        def best_stump(curr_data: np.ndarray, y: np.ndarray, stumps)->  np.ndarray:
            n_features = curr_data.shape[1]
            min_mistakes = n_samples + 1  # more mistakes that a a stump can reach
            best_stump = None
            for i in range(n_features):

                curr_mistakes = BaseEstimator.loss(stumps[i], curr_data, y)
                if curr_mistakes <= min_mistakes:
                    best_stump = stumps[i]
                    min_mistakes = curr_mistakes
            return best_stump

        # calculate the new distribution according to the best stump and it's weight
        def find_distribution(pred_labels, curr_labels, curr_learner):
            n_samples = curr_data.shape[0]
            # vector will contain the expression: Dit,j * e(-weight*y_j*f_j)
            D_e_vec = np.zeros(n_samples)
            sum = 0
            # filling in the dist_vector
            for j in range(n_samples):
                # the prediction of the current learner
                f_j = pred_labels[j]
                # true value of a label
                y_j = curr_labels[j]
                e_j = np.exp(-self.weights_[curr_learner] * y_j * f_j)
                D_e_vec[j] = self.D_[j]*e_j
                sum += self.D_[j]*e_j

            # calculating the new distribution
            for j in range(n_samples):
                self.D_[j] = D_e_vec[j] * 1/sum


        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.models_ = np.ndarray(self.iterations_, self.wl_)
        self.weights_ = np.zeros(self.iterations_)
        # sample distribution
        self.D_ = np.ndarray(n_samples)
        self.D_.fill(1/n_features)
        # create and fit stumps
        stumps = np.ndarray(shape=n_features, dtype=self.wl_)

        curr_data = X
        curr_labels = y
        for i in range(self.iterations_):
            # fitting stumps
            for k in range(n_features):
                stumps[k] = self.wl_()
                stumps[k].j_ = k
                stumps[k].fit(curr_data, curr_labels)

            # finding best stump and its weights
            self.models_[i] = best_stump(curr_data, curr_labels, stumps)
            pred_labels = BaseEstimator.predict(self.models_[i], curr_data)
            self.weights_[i] = find_weight(pred_labels, y, i)
            # recalculating distribution
            find_distribution(pred_labels, curr_labels, i)
            # resampling the data
            curr_data, curr_labels = resample(curr_data, curr_labels)

    def _predict(self, X):
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

        return self.partial_predict(X, self.iterations_)

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        # predicts the value of a sample with a weighted ensemble
        def estimator(sample, T):
            result = 0
            for i in range(T):
                result += self.weights_[i] * self.wl_.predict(self.models_[i], sample)

            return np.sign(result)

        # calculating weight vector
        # weight_vector = np.zeros(T)
        # for i in range(T):
        #     weight_vector[i] = find_weight(X, self.models_[i], i)

        # predicting every sample
        for i in range(n_samples):
            predictions[i] = estimator(X[i:i+1], T)

        return predictions

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        n_samples = X.shape[0]
        mistakes = 0
        for i in range(n_samples):
            if y[i] != y_pred[i]:
                mistakes += 1

        return mistakes
