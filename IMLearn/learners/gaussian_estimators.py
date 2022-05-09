from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = (1 / X.size) * X.sum()
        sum = 0
        for i in range(X.size):
            sum += (X[i] - self.mu_)**2
        if self.biased_:
            self.var_ = (1/X.size)*sum
        else:
            self.var_ = (1/(X.size - 1))*sum
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`pdf` function")
        num_of_samples = X.size
        values = np.zeros(num_of_samples)
        for i in range(num_of_samples):
            value = (1 / np.sqrt(2 * np.pi * self.var_)) * np.e ** (
                        (-0.5) * ((X[i] - self.mu_) / np.sqrt(self.var_)) ** 2)
            values[i] = value
        return values

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        num_of_samples = X.size
        sum = 0
        for x in X:
            # calculating the lof of the multiplication of the pdf values
            sum += -0.5*((x-mu)/sigma)**2
        result = -num_of_samples*(np.log(sigma)+0.5*np.log(2*np.pi)) + sum
        return result


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # important to remember: the rows represent the samples, the cols
        # represent the features
        # calculating estimated mean:
        n_features = X.shape[1]  # number of features
        n_samples = X.shape[0]  # number of samples
        mu_vector = np.zeros(n_features)
        for i in range(n_features):
            data_i = X[:, i:i+1]  # samples of only the i'th feature (column)
            # calculating the estimate of the i th feature, using
            # the univariate func we built before
            mu_vector[i] = UnivariateGaussian().fit(data_i).mu_

        self.mu_ = mu_vector
        #calculate var:

        def sigma_sum(i, j):
            sum = 0
            for k in range(n_samples):
                sum += (X[k, i] - self.mu_[i])*(X[k, j] - self.mu_[j])
            return sum

        covar_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                covar_matrix[i, j] = (1/n_samples)*(sigma_sum(i, j))

        self.cov_ = covar_matrix
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        n_samples = X.shape[1]  # number of samples
        func_vector = np.zeros(n_samples)
        for i in range(n_samples):
            i_degree = ((X[i, :] - self.mu_).transpose().dot(inv(self.cov_)).dot(X[i, :] - self.mu_))
            func_vector[i] = (1/np.linalg.det(self.cov_)(2*np.pi)**n_samples)*np.e**i_degree
        return func_vector

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]  # number of samples
        cov_det = np.linalg.det(cov)
        likelihood = 0
        for i in range(n_samples):
            sample_mu_dist = X[i, :] - mu.transpose()
            MD = (sample_mu_dist.transpose().dot(inv(cov)).dot(sample_mu_dist))
            f_xi = -0.5 * (n_features * np.log(2*np.pi) + np.log(cov_det) + MD)
            likelihood += f_xi
        return likelihood