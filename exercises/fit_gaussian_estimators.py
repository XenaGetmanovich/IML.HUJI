from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10.0, 1.0
    num_of_uni_samples = 1000
    data = np.random.normal(loc=mu, scale=sigma, size=num_of_uni_samples)
    values = UnivariateGaussian()
    values.fit(data)
    mean = values.mu_
    var = values.var_
    print("(", mean, ",", var, ")")

    # Question 2 - Empirically showing sample mean is consistent
    sample_step = 10
    step_range = np.arange(sample_step, num_of_uni_samples + sample_step, sample_step)
    mean_array = np.zeros(step_range.size)  # creating an array of expections
    # for different sets of samples
    for m in range(step_range.size):
        # for each number of samples m, calculating m
        m_data = data[0:step_range[m]]  # getting first m samples
        mean_array[m] = np.abs(mu - np.mean(m_data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = step_range, y = mean_array))
    fig.update_layout(title="Absolute distance between the estimated - and true value of the expectation as a function of the number of samples",xaxis_title="number of samples",
    yaxis_title="absolute distance",)
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    uni_gaus = UnivariateGaussian()
    uni_gaus.fit(data)
    pdf = uni_gaus.pdf(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data, y=pdf))
    fig.update_layout(
        title="Probabilty of the data acoording to the estimated expectation and variance",
        xaxis_title="samples",
        yaxis_title="probability", )
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    num_of_multi_samples = 1000
    mu = np.array([0, 0, 4, 0])
    sigma = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    # drawing samples cov=sigma, size=num_of_multi_samples
    data = np.random.multivariate_normal(mu, sigma, num_of_multi_samples)
    # fitting the samples
    obj = MultivariateGaussian()
    obj.fit(data)
    print(obj.mu_)
    print(obj.cov_)

    # Question 5 - Likelihood evaluation
    n = 15
    f1 = np.linspace(-10, 10, n)
    f3 = np.linspace(-10, 10, n)
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    # likelihood according to values in f1, f3
    multi_gauss = MultivariateGaussian()
    multi_gauss.fit(data)
    like_array = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mu_ij = np.array([f1[i], 0, f3[j], 0])
            like_array[i, j] = multi_gauss.log_likelihood(mu_ij, sigma, data)
    fig = px.imshow(like_array,
                    labels=dict(x="f3", y="f1", color="Likelihood"), title= "Heatmap of log-ikelihood according to f1 & f3")
    fig.show()

    # Question 6 - Maximum likelihood
    max = np.unravel_index(np.argmax(like_array), like_array.shape)
    # the max is in: f1[max[0]], f3(max[1])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

