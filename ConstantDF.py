import json
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm

from classes.DataReader import AggregatedDataReader
from classes.DiscountFactors import ConstantDF
from classes.ParamsEstimator import ParamsEstimatorBFGS
from classes.TrustEstimator import TrustEstimator
from classes.Utils import LearnerSettings, get_rmse

DISCOUNT_FACTOR = 1.0
LR1 = 0.01
MAX_ITERS = 1000
ERROR_TOL = 1e-6
TRAINING_LENGTH = 10
FEEDBACK_GAP = 5

LEARNER_SETTINGS = LearnerSettings(
    LR1,
    MAX_ITERS,
    ERROR_TOL,
    TRAINING_LENGTH,
    FEEDBACK_GAP,
    DISCOUNT_FACTOR,
    DISCOUNT_FACTOR,
    0.0,
)


def fit_data(
    df: pd.DataFrame, discount_factor: float, learner_settings: LearnerSettings
):
    """
    Runs BFGS to maximize the log-likelihood function of the observed feedback. The strategy is as follows:
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


def main():
    reader = AggregatedDataReader()
    reader.read_data()
    data = reader.data
    out_data = {
        "Participant ID": [],
        "Cluster": [],
        "Discount Factor": [],
        "Trust Estimate": [],
        "Trust Feedback": [],
        "rmse": [],
        "alpha0": [],
        "beta0": [],
        "ws": [],
        "wf": [],
        "True_state": [],
        "Alert": [],
        "Identification": [],
        "Performance": [],
    }

    participant_ids = data["Participant ID"].unique().tolist()

    for p_id in tqdm(participant_ids):
        df = data.loc[data["Participant ID"] == p_id]

        trust_estimates, trust_feedback, rmse, theta = fit_data(
            df, DISCOUNT_FACTOR, LEARNER_SETTINGS
        )

        out_data["Participant ID"].extend([p_id] * len(trust_feedback))
        out_data["Discount Factor"].extend([DISCOUNT_FACTOR] * len(trust_feedback))
        out_data["Trust Feedback"].extend(trust_feedback)
        out_data["Trust Estimate"].extend(list(trust_estimates))
        out_data["Cluster"].extend(df["Cluster"].values.tolist())
        out_data["alpha0"].extend(theta["alpha0"])
        out_data["beta0"].extend(theta["beta0"])
        out_data["ws"].extend(theta["ws"])
        out_data["wf"].extend(theta["wf"])
        out_data["rmse"].extend([rmse] * len(trust_feedback))
        out_data["True_state"].extend(df["True_state"].values.tolist())
        out_data["Alert"].extend((df["Alert"].values.tolist()))
        out_data["Identification"].extend((df["Identification"].values.tolist()))
        out_data["Performance"].extend(df["Performance"].values.tolist())

    out_data = pd.DataFrame(out_data)
    directory = os.path.join("AggregatedData", f"{DISCOUNT_FACTOR:.2f}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "BestEstimates.csv")
    out_data.to_csv(filename)
    settings_filepath = os.path.join(directory, "learner-settings.json")

    with open(settings_filepath, "w") as f:
        json.dump(LEARNER_SETTINGS.data, f, indent=4)


if __name__ == "__main__":
    main()
