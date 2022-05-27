#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(
    experiments_dir_path: Path,
):
    if not experiments_dir_path.is_dir():
        raise FileNotFoundError(
            f"{str(experiments_dir_path)} is not a valid directory!"
        )

    out_dir_path = experiments_dir_path / "plots"
    out_dir_path.mkdir(exist_ok=True)

    # Collect all the experiments directories with results
    line_plot_data = {"PQ": [], "mIoU": []}

    for metrics_file_path in experiments_dir_path.glob("**/metrics.csv"):
        metrics_df = (
            pd.read_csv(metrics_file_path)
            .sort_values(by=["MapID"])
            .drop(["MapID"], axis=1)  # Keep only metrics to plot
        )

        # Insert TimeStamp column
        metrics_df.insert(
            0,
            "TimeStamp",
            np.arange(
                60,
                (len(metrics_df.index) + 1) * 60,
                step=60,
            ),
        )

        # Insert new column to indicate method used in the experiment
        # (it should be the experiment name)
        experiment_name = metrics_file_path.parent.name
        metrics_df["Method"] = experiment_name

        line_plot_data["PQ"].append(metrics_df[["Method", "TimeStamp", "PQ"]])
        line_plot_data["mIoU"].append(metrics_df[["Method", "TimeStamp", "mIoU"]])

    # Now for each metric
    sns.set_style("darkgrid")
    for metric_name, plot_data in line_plot_data.items():
        plot_data_df = pd.concat(plot_data, axis=0)
        long_format_plot_data_df = pd.melt(
            plot_data_df,
            id_vars=["TimeStamp", "Method"],
            value_name=metric_name,
        )

        sns.lineplot(
            x="TimeStamp",
            y=metric_name,
            hue="Method",
            data=long_format_plot_data_df,
        )

        plot_file_path = out_dir_path / f"{metric_name}.png"
        plt.savefig(str(plot_file_path))
        plt.clf()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate line plots of planning experiments results."
    )

    parser.add_argument(
        "experiments_dir",
        type=lambda p: Path(p).absolute(),
        help="Path to the directory contanining the experiments results.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.experiments_dir)
