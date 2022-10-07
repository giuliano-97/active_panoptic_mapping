#!/usr/bin/env python3

import argparse
import logging
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from active_panoptic_mapping_utils.evaluation.constants import (
    NYU40_CLASSES,
    NYU40_THING_CLASSES,
    NYU40_CLASS_IDS_TO_NAMES,
    SCANNET_NYU40_EVALUATION_CLASSES,
    TP_FP_FN_KEYS,
)

plt.ioff()


def _remove_vector_values(d: Mapping):
    result = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            result[k] = _remove_vector_values(v)
        elif isinstance(v, Sequence):
            continue
        else:
            result[k] = v
    return result


def _extract_metrics_regex(metrics_by_method, regex=".*"):
    matcher = re.compile(regex)
    result = defaultdict(dict)
    for method, metrics in metrics_by_method.items():
        for k,v in metrics.items():
            if matcher.match(k):
                result[method][k] = v
    return dict(result)


def _plot_confusion_matrix(confusion_matrix: np.ndarray, display_labels: List, output_file_path: Path):
    confusion_matrix_only_eval_classes = confusion_matrix[[0] + SCANNET_NYU40_EVALUATION_CLASSES, :][:, [0] + SCANNET_NYU40_EVALUATION_CLASSES]
    fig, ax = plt.subplots(1, 1, figsize=(30, 20))

    ConfusionMatrixDisplay(
        confusion_matrix_only_eval_classes,
        display_labels=display_labels,
    ).plot(ax=ax)

    plt.savefig(str(output_file_path), bbox_inches="tight")
    plt.clf()


def plot_mapping_metrics(metrics: Dict, output_dir_path: Path):

    display_labels = ["ignore"] + [NYU40_CLASS_IDS_TO_NAMES[i] for i in SCANNET_NYU40_EVALUATION_CLASSES]

    # Plot confusion matrices

    for method, method_metrics in metrics.items():
        method_plots_dir = output_dir_path / method
        method_plots_dir.mkdir(parents=True, exist_ok=True)

        sem_seg_cmat = np.array(method_metrics["semantic_confusion_matrix"])
        _plot_confusion_matrix(sem_seg_cmat, display_labels, method_plots_dir / "sem_seg_cmat.png")
        
        matched_segments_cmat = np.array(method_metrics["matched_segments_confusion_matrix"])
        _plot_confusion_matrix(matched_segments_cmat, display_labels, method_plots_dir / "sem_seg_cmat.png")


    # Plot scalar metrics as bar plots
    metrics_scalar_only = _remove_vector_values(metrics)
    for prefix in ["PQ", "SQ", "RQ", "FP", "FN"]:
        metrics_scalar_only_filtered = _extract_metrics_regex(metrics_scalar_only, f"^{prefix}.*")
        metrics_df = pd.DataFrame.from_dict(metrics_scalar_only_filtered, orient='columns').fillna(0)
        metrics_df.plot.barh()
        plt.savefig(str(output_dir_path / f"{prefix}.png"))
        plt.clf()
    

    for prefix in ["PQ", "SQ", "RQ", "FP", "FN", "iou"]:
        metrics_scalar_only_filtered = _extract_metrics_regex(metrics_scalar_only, f"^{prefix}.*")
        metrics_df = pd.DataFrame.from_dict(metrics_scalar_only_filtered, orient='columns').fillna(0)
        metrics_df.plot.barh()
        plt.savefig(str(output_dir_path / f"{prefix}.png"))
        plt.clf()

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and plot results of mapping experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "experiments_dir",
        type=lambda p: Path(p).expanduser().resolve(),
        help="Path to the directory containing the experiments.")

    parser.add_argument(
        "-o",
        "--output-dir",
        type=lambda p: Path(p).expanduser().resolve(),
        help="Path to the output directory.")
 
    return parser.parse_args()
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args =_parse_args()

    metrics_file_path = args.experiments_dir / "metrics.json"
    if not metrics_file_path.is_file():
        logging.error("Metrics file does not exist.")
        exit(1)
    
    output_dir_path = args.output_dir
    if output_dir_path is None:
        output_dir_path = args.experiments_dir / "plots"
        output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_file_path, "r") as f:
        metrics = json.load(f)

    plot_mapping_metrics(metrics, output_dir_path)
    

    