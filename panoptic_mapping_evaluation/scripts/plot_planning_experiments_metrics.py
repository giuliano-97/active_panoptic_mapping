#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


_VALUE_VARS = ["PQ", "mIoU"]
_ID_VARS = ["Method", "TimeStamp"]


def main(
    experiments_dir_path: Path,
):
    if not experiments_dir_path.is_dir():
        raise FileNotFoundError(
            f"{str(experiments_dir_path)} is not a valid directory!"
        )

    out_dir_path = experiments_dir_path / "plots"
    out_dir_path.mkdir(exist_ok=True)

    # Collect all the metrics
    metrics_data = []
    for metrics_file_path in experiments_dir_path.glob("**/metrics.csv"):
        metrics_df = pd.read_csv(metrics_file_path).sort_values(by=["Method", "MapID"])
        metrics_df["TimeStamp"] = metrics_df.apply(lambda L: int(L.MapID), axis=1)
        metrics_data.append(metrics_df)

    # Now concatenate
    metrics_data_df = pd.concat(metrics_data, axis=0)

    # Now for each metric
    sns.set_style("darkgrid")
    for metric in _VALUE_VARS:
        plot_data_df = metrics_data_df[_ID_VARS + [metric]]
        long_format_plot_data_df = pd.melt(
            plot_data_df,
            id_vars=["TimeStamp", "Method"],
            var_name="metric",
            value_name="value",
        )

        # Create new figure
        plt.figure(figsize=(16, 9))

        sns.lineplot(
            x="TimeStamp",
            y="value",
            hue="Method",
            data=long_format_plot_data_df,
            estimator="mean",
            ci="sd",
            err_style="band",
        )

        plot_file_path = out_dir_path / f"{metric}.png"
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
