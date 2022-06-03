import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# DELETE #
import time
start_time = time.time()



def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_learner = AdaBoost(DecisionStump, n_learners)
    AdaBoost.fit(ada_learner, train_X, train_y)
    train_mistakes = np.zeros(n_learners)
    test_mistakes = np.zeros(n_learners)
    mistakes = np.zeros((n_learners, 2))
    for i in range(n_learners):
        train_mistakes[i] = ada_learner.partial_loss(train_X, train_y, i)
        test_mistakes[i] = ada_learner.partial_loss(test_X, test_y, i)

    # plotting the graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n_learners), y=train_mistakes, name="train set"))
    fig.add_trace(go.Scatter(x=np.arange(n_learners), y=test_mistakes, name="test set"))
    fig.update_layout(
        title="train set and test set mistakes as a function of the number of the learners",
        xaxis_title="number of mistakes",
        yaxis_title="number of learners", )
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    #
    # # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()
    #
    # # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # stumps = np.ndarray(5, DecisionStump)
    # a = DecisionStump()
    # stumps.fill(DecisionStump())
    #
    print("--- %s seconds ---" % (time.time() - start_time))
    # raise NotImplementedError()
