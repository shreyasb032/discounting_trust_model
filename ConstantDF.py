import json
import os.path

import pandas as pd
from tqdm import tqdm

from classes.DataReader import AggregatedDataReader
from classes.Utils import LearnerSettings, fit_data

DISCOUNT_FACTOR = 0.8
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
