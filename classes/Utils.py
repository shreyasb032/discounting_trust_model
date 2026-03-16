import numpy as np
import pandas as pd

from classes.DiscountFactors import ConstantDF
from classes.ParamsEstimator import ParamsEstimatorBFGS
from classes.TrustEstimator import TrustEstimator


class LearnerSettings:
    """
    Class to store the settings of the parameter learner class
    """

    def __init__(
        self,
        lr: float,
        max_iters: int,
        error_tol: float,
        training_length: int,
        feedback_gap: int,
        df_start: float | None,
        df_end: float | None,
        df_stepsize: float | None,
        lr3: float | None = None,
    ):
        """
        :param lr: the learning rate for gradient ascent for ws, wf
        :param max_iters: the maximum number of iterations of gradient descent to run whenever new data is added
        :param error_tol: the tolerance in the magnitude of gradients below which the gradient ascent algorithm is
                          stopped
        :param training_length: the number of interactions to be used as a training set
        :param feedback_gap: the gap between consecutive trust feedbacks after the training set is completed
        :param df_start: the value at which to start the discount factor grid
        :param df_end: the value at which to end the discount factor grid
        :param df_stepsize: the stepsize of the discount factor grid
        :param lr3: the learning rate for the lambda parameter if used
        """

        self.lr = lr
        self.max_iters = max_iters
        self.error_tol = error_tol
        self.training_length = training_length
        self.feedback_gap = feedback_gap
        self.df_start = df_start
        self.df_end = df_end
        self.df_stepsize = df_stepsize
        self.lr3 = lr3

        self.data = {
            "Learning rate ws wf": lr,
            "Max iterations": max_iters,
            "Error tolerance": error_tol,
            "Training length": training_length,
            "Feedback gap": feedback_gap,
            "Discount factor": {
                "Start": df_start,
                "End": df_end,
                "Stepsize": df_stepsize,
            },
        }

        if lr3 is not None:
            self.data["Learning rate lambda"] = lr3


def get_rmse(x: np.ndarray, y: np.ndarray):
    """
    Returns the RMSE between two 1D numpy arrays. Make sure that the lengths of the arrays match
    :param x: a numpy 1D array
    :param y: a numpy 1D array
    """

    return np.sqrt(np.dot(x - y, x - y) / x.shape[0])


def fit_data(
    df: pd.DataFrame, discount_factor: float, learner_settings: LearnerSettings
):
    """
    Runs gradient ascent to maximize the log-likelihood function of the observed feedback. The strategy is as follows:
    Treat each feedback point as a new data in a time series. Fit a model to the time series data until the current data
    point. Estimate the trust at current trial number using the fitted data. Store the model and the estimated trust
    data.
    :param df: a pandas dataframe with keys ['Performance', 'Trust']
    :param discount_factor: the constant discount factor to be used for data fitting.
    :param learner_settings: the settings for the parameter learner class
    """

    _discount_factor = ConstantDF(discount_factor=discount_factor)
    trust_params = {"alpha0": None, "beta0": None, "ws": None, "wf": None}
    trust_estimator = TrustEstimator(_discount_factor, trust_params)
    params_estimator = ParamsEstimatorBFGS(discount_factor=_discount_factor)

    num_sites = len(df.index)
    training_length = learner_settings.training_length
    feedback_gap = learner_settings.feedback_gap

    params_all = {
        "alpha0": [],
        "beta0": [],
        "ws": [],
        "wf": [],
        "alpha": [],
        "beta": [],
    }
    performance = df["Performance"].to_numpy()
    trust_feedback = df["Trust"].to_numpy() / 100.0
    trust_estimates = np.zeros_like(trust_feedback)

    for i in range(num_sites):
        if i < training_length or (i + 1) % feedback_gap == 0:
            trust_params = params_estimator.estimate(
                trust_feedback[i], performance[i], trust_params
            )
        else:
            params_estimator.add_performance(performance[i])

        for k, v in trust_params.items():
            params_all[k].append(v)

        trust_estimator.update_params(trust_params)
        trust_estimate, alpha, beta = trust_estimator.get_trust(performance[i], i)
        trust_estimates[i] = trust_estimate
        params_all["alpha"].append(alpha)
        params_all["beta"].append(beta)

    rmse = get_rmse(trust_estimates[training_length:], trust_feedback[training_length:])

    return trust_estimates, trust_feedback, rmse, params_all
