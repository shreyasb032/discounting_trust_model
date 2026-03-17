import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from classes.DataReader import AggregatedDataReader
from classes.Utils import LearnerSettings, fit_data

ORDER = 0
LR = 0.01
MAX_ITERS = 1000
ERROR_TOL = 1e-6
TRAINING_LENGTH = 10
FEEDBACK_GAP = 5
STEP_SIZE = 0.01
START = 0.1
END = 1.0

LEARNER_SETTINGS = LearnerSettings(
    LR, MAX_ITERS, ERROR_TOL, TRAINING_LENGTH, FEEDBACK_GAP, START, END, STEP_SIZE
)


def main():
    # Each time the code is run, create a new folder to store the data
    directory = os.path.join("AggregatedData", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_steps = round((END - START) / STEP_SIZE) + 1
    discount_factors = np.linspace(START, END, num_steps)

    reader = AggregatedDataReader()
    reader.read_data()
    data = reader.data

    participant_ids = data["Participant ID"].unique().tolist()

    for p_id in participant_ids:
        filename = os.path.join(directory, f"Participant{p_id:03}.xlsx")

        df = data.loc[data["Participant ID"] == p_id]
        all_sheets_data = []

        for discount_factor in tqdm(discount_factors, desc=f"Participant {p_id:03}"):
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
                "alpha": [],
                "beta": [],
                "True_state": [],
                "Alert": [],
                "Identification": [],
                "Performance": [],
            }

            trust_estimates, trust_feedback, rmse, theta = fit_data(
                df, discount_factor, LEARNER_SETTINGS
            )
            data_len = len(trust_feedback)
            out_data["Participant ID"] = [p_id] * data_len
            out_data["Cluster"] = df["Cluster"].values.tolist()
            out_data["Discount Factor"] = [discount_factor] * data_len
            out_data["Trust Estimate"] = list(trust_estimates)
            out_data["Trust Feedback"] = list(trust_feedback)
            out_data["rmse"] = [rmse] * data_len
            for k, v in theta.items():
                out_data[k] = v
            out_data["True_state"] = df["True_state"].values.tolist()
            out_data["Alert"] = df["Alert"].values.tolist()
            out_data["Identification"] = df["Identification"].values.tolist()
            out_data["Performance"] = df["Performance"].values.tolist()

            out_df = pd.DataFrame(out_data)
            all_sheets_data.append(out_df)

        # Write consolidated data to single sheet
        if all_sheets_data:
            consolidated_df = pd.concat(all_sheets_data, ignore_index=True)
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                consolidated_df.to_excel(writer, sheet_name="Results", index=False)

    settings_filepath = os.path.join(directory, "learner-settings.json")
    with open(settings_filepath, "w") as f:
        json.dump(LEARNER_SETTINGS.data, f, indent=4)


if __name__ == "__main__":
    main()
