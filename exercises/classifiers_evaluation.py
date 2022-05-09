from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # we load the data from each one
        file_name = "C:\\Users\\Xen\Documents\\University\\IML\\GIT IML\\IML.HUJI\\datasets\\" + f
        print(file_name)
        data = np.load(file_name)
        array = np.zeros(3)
        n_samples = data.shape[0]
        n_features = data.shape[1] - 1
        X = data[:, :n_features]
        y_true = data[:, n_features:]

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_recorder(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit._loss(X, y_true))

        # create Perceptron object
        new_perc = Perceptron(callback=loss_recorder)
        new_perc.fit(X,y_true)

        # Plot figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(losses))), y=losses))
        fig.update_layout(
            title="Losses over number of iterations",
            xaxis_title="iterations",
            yaxis_title="losses", )
        fig.show()

        # raise NotImplementedError()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        raise NotImplementedError()

        # Fit models and predict over training set
        raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # compare_gaussian_classifiers()
