#!/usr/bin/env python3

import argparse
from pathlib import Path
from git import BadName

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_EVALUATION_METRICS = ["PQ", "mIoU"]
_OTHER_METRICS = ["Coverage"]

_X_AXIS_NAME = "Simulated Time [min]"

_METRIC_TO_Y_AXIS_NAME = {
    "PQ": "Observed Surface Panoptic Quality [%]",
    "mIoU": "Observed Surface mIoU [%]",
    "Coverage": "Observed Surface [%]",
}

_METRIC_TO_TITLE = {
    "PQ": "Panoptic Quality",
    "mIoU": "Mean IoU",
    "Coverage": "Exploration Rate",
}
_HUE_NAME = "Method"

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
        metrics_df[_X_AXIS_NAME] = metrics_df.apply(lambda L: int(L.MapID) // 60, axis=1)
        metrics_data.append(metrics_df)

    # Now concatenate
    metrics_data_df = pd.concat(metrics_data, axis=0)
    metrics_data_df[_EVALUATION_METRICS + _OTHER_METRICS] *= 100

    # Now for each metric
    # sns.set_style("darkgrid")
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set(font_scale=1.5)
    for metric in _EVALUATION_METRICS + _OTHER_METRICS:
        plot_data_df = metrics_data_df[[_HUE_NAME, _X_AXIS_NAME, metric]]
        value_name = _METRIC_TO_Y_AXIS_NAME[metric]
        long_format_plot_data_df = pd.melt(
            plot_data_df,
            id_vars=[_X_AXIS_NAME, _HUE_NAME],
            value_name=value_name,
        )

        # Create new figure
        fig = plt.figure(figsize=(16, 9))
        plt.title(_METRIC_TO_TITLE[metric])
        plt.ylim(0, 100.0)

        sns.lineplot(
            x=_X_AXIS_NAME,
            y=value_name,
            hue=_HUE_NAME,
            data=long_format_plot_data_df,
            estimator="mean",
            ci="sd",
            err_style="band",
        )

        plot_file_path = out_dir_path / f"{metric}.png"
        plt.savefig(str(plot_file_path), bbox_inches="tight")
        plt.close(fig)
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

