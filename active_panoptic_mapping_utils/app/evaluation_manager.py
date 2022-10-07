import json
from collections.abc import Mapping
from collections import defaultdict
from pathlib import Path
from re import S
from typing import List

import numpy as np
import rospy
from tqdm import tqdm

from active_panoptic_mapping_utils.srv import ExportEvaluationData

from active_panoptic_mapping_utils.evaluation.constants import (
    NYU40_IGNORE_LABEL,
    SCANNET_NYU40_EVALUATION_CLASSES,
    PANOPTIC_LABEL_DIVISOR,
)
from active_panoptic_mapping_utils.evaluation.metrics import (
    panoptic_quality,
    PanopticQuality,
    mean_iou,
    MeanIoU,
    coverage,
    Compose,
    compose,
)

def make_dict_json_serializable(d):
    sd = {}
    for k, v in d.items():
        if isinstance(v, (np.ndarray, np.generic)):
            if np.issubdtype(v.dtype, np.integer):
                v = v.astype(int)
            elif np.issubdtype(v.dtype, np.floating):
                v = v.astype(float)
            else:
                raise ValueError("unsupported numpy dtype in metrics")                    

            if isinstance(v, np.ndarray):
                sd[k] = v.tolist()
            elif isinstance(v, np.generic):
                sd[k] = v.item()
        elif isinstance(v, Mapping):
            sd[k] = make_dict_json_serializable(v)
    return sd

def mask_ignored_categories(vertex_panoptic_labels: np.ndarray):
    res = np.copy(vertex_panoptic_labels)
    category_ids = vertex_panoptic_labels // PANOPTIC_LABEL_DIVISOR
    for c in np.unique(category_ids):
        if c not in SCANNET_NYU40_EVALUATION_CLASSES:
            res[category_ids == c] = NYU40_IGNORE_LABEL
    return res


class EvaluationManager:
    def __init__(self):
        rospy.init_node("evaluation_manager_node")

        self.experiments_dir_path = Path(rospy.get_param("~experiments_dir"))
        self.gt_dir_path = Path(rospy.get_param("~ground_truth_dir"))

        self.experiment_type = rospy.get_param("~experiment_type", "mapping").lower()

        if self.experiment_type not in ["mapping", "planning"]:
            raise ValueError("Invalid experiment type!")

        self.export_eval_data_srv_name = rospy.get_param(
            "~export_eval_data_srv_name",
            "/export_evaluation_data",
        )

        if not self.experiments_dir_path.is_dir():
            rospy.logfatal(f"{self.experiments_dir_path} is not a directory!")

        rospy.wait_for_service(self.export_eval_data_srv_name)
        self.export_eval_data_srv_proxy = rospy.ServiceProxy(
            self.export_eval_data_srv_name,
            ExportEvaluationData,
        )

    def _get_map_gt_dir(self, map_file_path: Path):
        map_gt_dir_name = (
            map_file_path.parents[2].name
            if self.experiment_type == "planning"
            else map_file_path.parents[1].name
        )
        return self.gt_dir_path / map_gt_dir_name

    def _get_map_gt_pointcloud_file(self, map_file_path: Path):
        gt_dir_path = self._get_map_gt_dir(map_file_path)
        if self.experiment_type == "mapping":
            return gt_dir_path / f"{gt_dir_path.name}_vh_clean_2.ply"
        elif self.experiment_type == "planning":
            raise NotImplementedError

    def _get_map_gt_vertex_labels_file(self, map_file_path: Path):
        gt_dir_path = self._get_map_gt_dir(map_file_path)
        return gt_dir_path / "panoptic_vertex_labels.txt"

    def _export_evaluation_data(self):
        map_files = sorted(list(self.experiments_dir_path.glob("**/*.panmap")))

        pred_vertex_labels_files = []

        for map_file_path in map_files:
            pred_vertex_labels_file_path = map_file_path.with_suffix(".txt")

            if not pred_vertex_labels_file_path.is_file():
                gt_pointcloud_file_path = self._get_map_gt_pointcloud_file(
                    map_file_path
                )
                if not gt_pointcloud_file_path.is_file():
                    rospy.logerror("Ground truth file not found. Skipped.")
                    continue

                rospy.loginfo(f"Exporting evaluation data for {map_file_path.name}")
                try:
                    _ = self.export_eval_data_srv_proxy(
                        str(gt_pointcloud_file_path.absolute()),
                        str(map_file_path.absolute()),
                    )
                except rospy.ServiceException as se:
                    rospy.logerr(
                        f"Export request failed with exception: {se}. Skipped."
                    )
                    continue

            pred_vertex_labels_files.append(pred_vertex_labels_file_path)

        return pred_vertex_labels_files

    def _load_gt_and_pred_vertex_labels(
        self,
        gt_vertex_labels_file_path: Path,
        pred_vertex_labels_file_path: Path,
    ):
        # Load gt and predicted vertex panoptic labels
        # FIXME: should we mask here? it's already done in the metrics
        gt_vertex_labels = mask_ignored_categories(
            np.loadtxt(str(gt_vertex_labels_file_path)).astype(np.int64)
        )
        pred_panoptic_vertex_labels = np.loadtxt(
            str(pred_vertex_labels_file_path)
        ).astype(np.int64)

        # Sanity checks
        if len(pred_panoptic_vertex_labels) != len(gt_vertex_labels):
            rospy.logwarn(
                "Predicted and ground truth vertex labels do not have the same shape."
            )
            return None, None

        # Compute coverage mask
        covered_vertices_mask = pred_panoptic_vertex_labels != -1

        # Mask out vertices that were not covered and should not be evaluated
        covered_gt_vertex_labels = np.where(
            covered_vertices_mask, gt_vertex_labels, NYU40_IGNORE_LABEL
        )

        # Remove the -1 representing covered vertices from prediction
        rectified_pred_vertex_labels = np.where(
            covered_vertices_mask, pred_panoptic_vertex_labels, NYU40_IGNORE_LABEL
        )

        if self.experiment_type == "mapping":
            return covered_gt_vertex_labels, rectified_pred_vertex_labels
        else:
            return gt_vertex_labels, rectified_pred_vertex_labels

    def _evaluate_mapping_experiments(self, pred_vertex_labels_files: List[Path]):
        pred_vertex_labels_files_by_method = defaultdict(list)
        for p in pred_vertex_labels_files:
            method = p.parent.name
            pred_vertex_labels_files_by_method[method].append(p)

        metrics_by_method = {}

        for (
            method_name,
            method_pred_vertex_labels_files,
        ) in pred_vertex_labels_files_by_method.items():
            mapping_metrics = Compose([PanopticQuality(), MeanIoU()])
            for pred_vertex_labels_file_path in method_pred_vertex_labels_files:
                gt_vertex_labels_file_path = self._get_map_gt_vertex_labels_file(
                    pred_vertex_labels_file_path.with_suffix(".panmap")
                )

                if not gt_vertex_labels_file_path.is_file():
                    rospy.logerr(
                        f"Ground truth labels not found for {str(pred_vertex_labels_file_path)}."
                        " Skipped."
                    )
                    continue

                (
                    gt_vertex_labels,
                    pred_vertex_labels,
                ) = self._load_gt_and_pred_vertex_labels(
                    gt_vertex_labels_file_path,
                    pred_vertex_labels_file_path,
                )

                if gt_vertex_labels is None or pred_vertex_labels is None:
                    continue

                mapping_metrics.update(gt_vertex_labels, pred_vertex_labels)
            metrics_by_method[method_name] = mapping_metrics.compute()

        return metrics_by_method

    def _evaluate_planning_experiments(self, pred_vertex_labels_files: List[Path]):

        eval_planning_metrics_fn = compose([panoptic_quality, mean_iou, coverage])

        pred_vertex_labels_sequences_by_method = defaultdict(lambda: defaultdict(list))
        for p in pred_vertex_labels_files:
            run_name = p.parent.name
            method_name = p.parent.parents[2].name
            pred_vertex_labels_sequences_by_method[method_name][run_name].append(p)

        metrics_by_method = {}
        for (
            method_name,
            pred_vertex_labels_sequences,
        ) in pred_vertex_labels_sequences_by_method.items():
            method_metrics = []
            for (
                run_name,
                run_pred_vertex_labels_files,
            ) in pred_vertex_labels_sequences.items():
                run_pred_vertex_labels_files.sort()
                run_metrics = {}
                for pred_vertex_labels_file_path in run_pred_vertex_labels_files:
                    gt_vertex_labels_file_path = self._get_map_gt_vertex_labels_file(
                        pred_vertex_labels_file_path.with_suffix(".panmap")
                    )
                    (
                        pred_vertex_labels,
                        gt_vertex_labels,
                    ) = self._load_gt_and_pred_vertex_labels(
                        gt_vertex_labels_file_path,
                        pred_vertex_labels_file_path,
                    )

                    if gt_vertex_labels is None or pred_vertex_labels is None:
                        continue

                    map_id = pred_vertex_labels_file_path.stem
                    run_metrics[map_id] = eval_planning_metrics_fn(
                        gt_vertex_labels,
                        pred_vertex_labels,
                    )

                method_metrics.append(run_metrics)
            metrics_by_method[method_name] = method_metrics
        return metrics_by_method

    def evaluate(self):
        pred_vertex_labels_files = self._export_evaluation_data()

        if self.experiment_type == "mapping":
            metrics_by_method = self._evaluate_mapping_experiments(
                pred_vertex_labels_files
            )
        elif self.experiment_type == "planning":
            metrics_by_method = self._evaluate_planning_experiments(
                pred_vertex_labels_files
            )

        serializable_metrics_by_method = make_dict_json_serializable(metrics_by_method)
            

        metrics_file_path = self.experiments_dir_path / "metrics.json"
        with open(metrics_file_path, "w") as f:
            json.dump(serializable_metrics_by_method, f, indent=4)


if __name__ == "__main__":
    evaluation_manager = EvaluationManager()
    evaluation_manager.evaluate()
