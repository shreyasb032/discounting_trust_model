import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="talk", style="white")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="BestEstimates",
        help="path to file with generated trust data",
    )

    args = parser.parse_args()

    filepath = args.path
    df = pd.read_csv(filepath)
    palette = sns.color_palette()

    participant_ids = df["Participant ID"].unique().tolist()

    for p_id in participant_ids:
        participant_df = df.loc[df["Participant ID"] == p_id]

        trust_estimates = participant_df["Trust Estimate"].to_numpy()
        trust_feedback = participant_df["Trust Feedback"].to_numpy()
        cluster = participant_df["Cluster"].values.tolist()[0]
        discount_factor = participant_df["Discount Factor"].values.tolist()[0]

        fig, ax = plt.subplots(layout="tight")
        ax.plot(
            np.arange(trust_feedback.shape[0]),
            trust_feedback,
            lw=2,
            c=palette[0],
            label="Feedback",
        )
        ax.plot(
            np.arange(trust_feedback.shape[0]),
            trust_estimates,
            lw=2,
            c=palette[1],
            label="Estimate",
        )
        ax.legend()
        ax.set_title(rf"P{p_id}, {cluster}, $\lambda= {discount_factor:.1f}$")
        ax.set_ylim((-0.05, 1.05))
        ax.set_xlabel("Number of Interactions")
        ax.set_ylabel("Trust")
        directory = os.path.join("images", args.path, cluster)
        if not os.path.exists(directory):
            os.makedirs(directory)

        out_path = os.path.join(directory, f"Participant{p_id:03}.png")
        plt.savefig(out_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
