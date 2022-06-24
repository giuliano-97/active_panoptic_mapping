#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.ioff()

_EVALUATION_METRICS = ["PQ", "PQ_th", "PQ_st", "SQ", "RQ", "mIoU"]
_X_AXIS_NAME = "Metric"
_Y_AXIS_NAME = "Value (%)"


def main(experiments_dir_path: Path):
    # Collect metric files
    metrics_files = experiments_dir_path.glob("**/metrics.csv")

    out_dir_path = experiments_dir_path / "plots"
    out_dir_path.mkdir(exist_ok=True, parents=True)

    # Read metrics from files
    metrics_data = [
        pd.read_csv(p)[["Method"] + _EVALUATION_METRICS] for p in metrics_files
    ]

    # Aggregate results by method and compute the mean
    aggregated_metrics_df = (
        pd.concat(metrics_data, axis=0).groupby("Method").mean().reset_index()
    )

    # Convert values to percentages
    aggregated_metrics_df[_EVALUATION_METRICS] *= 100

    sns.set_style("darkgrid")
    sns.set(font_scale=1.5)
    for metrics_subset in [["PQ", "SQ", "RQ", "mIoU"]]:
        aggregated_metrics_subset_df_long = pd.melt(
            aggregated_metrics_df[["Method"] + metrics_subset],
            id_vars=["Method"],
            var_name=_X_AXIS_NAME,
            value_name=_Y_AXIS_NAME,
        )

        # Create new figure
        plt.figure(figsize=(16, 9))
        plt.ylim(0.0, 100.0)

        sns.barplot(
            x=_X_AXIS_NAME,
            y=_Y_AXIS_NAME,
            hue="Method",
            data=aggregated_metrics_subset_df_long,
        )

        plot_file_path = out_dir_path / f"{'_'.join(metrics_subset)}.png"
        plt.savefig(str(plot_file_path), bbox_inches="tight")
        plt.clf()


def _parse_args():
    parser = argparse.ArgumentParser("Plot evaluation metrics of mapping experiments.")

    parser.add_argument(
        "experiments_dir",
        type=lambda p: Path(p).resolve(strict=True),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.experiments_dir)