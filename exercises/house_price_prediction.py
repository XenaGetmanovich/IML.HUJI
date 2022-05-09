from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # load data as a dataframe
    df = pd.read_csv(filename)
    prices = df['price']
    print(df.shape)
    n_samples = df.shape[0]
    n_features = df.shape[1]
    # remove samples with damaged information
    damaged_samples = []
    # price too low
    # room sizes negative or too small < 11 sqft
    # lot too small or negaitve <20

    # for i in range(n_samples):
    #     if df.at[i, 'price'] < 1000 or df.at[i, 'sqft_lot'] < 11 or \
    #             df.at[i, 'sqft_living'] < 20:
    #         damaged_samples.append(i)

    # remove unnecessary values
    # tmp_df = df.drop(['id', 'date', 'price', 'lat', 'long'], axis=1)
    tmp_df = df.drop(['price'], axis=1)
    # clean_df = tmp_df.drop(damaged_samples, axis=0)
    print(tmp_df.shape)
    return tmp_df, prices
    # raise NotImplementedError()


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    # appending the sample array with the reault
    X_arr = X.to_numpy()
    X_with_res = np.c_[X_arr, y]
    print(type(X_with_res))
    print(X.shape)
    # for each feature:
    cov_mat = np.cov(X_with_res)
    y_std = np.std(y)
    for i in range(n_features):
        # calc the diviation
        feature_vec = X_arr[:, i: i+1]
        feature_std = np.std(feature_vec)
        pea_corr = (cov_mat[i, n_features+1])/(feature_std*y_std)

        #   plot the graph (feature name / result)
        feature_name = X.at[0, i]
        title = "Pearson Correlation between" + feature_name + "and response:" + pea_corr
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feature_vec, y=y))
        fig.update_layout(
            title=title,
            xaxis_title="samples",
            yaxis_title="results", )
        fig.show() # remove

        # save the graph in the giver path with the name
        file_name = feature_name + "correlation to results"
        fig.write_image(output_path+"/"+file_name)
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    str = r"C:\Users\Xen\Documents\University\IML\GIT IML\IML.HUJI\datasets\house_prices.csv"
    Data, y = load_data(str)
    # raise NotImplementedError()
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(Data, y)

    # raise NotImplementedError()
    #
    # # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()
    #
    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
