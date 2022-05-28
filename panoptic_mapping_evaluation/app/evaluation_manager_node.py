from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import rospy

from panoptic_mapping_msgs.srv import SaveLoadMap

from panoptic_mapping_evaluation.constants import NYU40_IGNORE_LABEL
from panoptic_mapping_evaluation.metrics import panoptic_quality, mean_iou


class EvaluationManager:
    def __init__(self):
        rospy.init_node("evaluation_manager_node")

        self.data_dir_path = Path(rospy.get_param("~data_dir"))
        self.experiment_type = rospy.get_param("~experiment_type", "mapping").lower()

        if self.experiment_type not in ["mapping", "planning"]:
            raise ValueError("Invalid experiment type!")

        self.gt_vertex_labels_file_path = Path(
            rospy.get_param("~ground_truth_vertex_labels_file")
        )
        self.export_eval_data_srv_name = rospy.get_param(
            "~export_eval_data_srv_name",
            "/panoptic_mapping_evaluation/export_evaluation_data",
        )

        if not self.data_dir_path.is_dir():
            rospy.logfatal(f"{self.data_dir_path} is not a directory!")

        if not self.gt_vertex_labels_file_path.is_file():
            rospy.logfatal(f"{self.gt_vertex_labels_file_path} is not a valid file!")

        # Load gt vertex labels
        self.gt_vertex_labels = np.loadtxt(self.gt_vertex_labels_file_path).astype(
            np.int64
        )

        rospy.wait_for_service(self.export_eval_data_srv_name)
        self.export_eval_data_srv_proxy = rospy.ServiceProxy(
            self.export_eval_data_srv_name,
            SaveLoadMap,
        )

    def evaluate(self):
        map_files = sorted(list(self.data_dir_path.glob("**/*.panmap")))

        metrics_list = []

        for map_file_path in map_files:
            rospy.loginfo(f"Exporting evaluation data for {map_file_path.name}")
            success = self.export_eval_data_srv_proxy(str(map_file_path.absolute()))
            if not success:
                rospy.logerror(f"Skipped.")

            pred_vertex_labels_file_path = map_file_path.with_suffix(".txt")
            if not pred_vertex_labels_file_path.is_file():
                rospy.logwarn(
                    f"{str(pred_vertex_labels_file_path)} not found. Skipped."
                )
                continue

            # Load predicted vertex panoptic labels
            pred_panoptic_vertex_labels = np.loadtxt(
                str(pred_vertex_labels_file_path)
            ).astype(np.int64)

            # Sanity checks
            if len(pred_panoptic_vertex_labels) != len(self.gt_vertex_labels):
                rospy.logwarn(
                    "Predicted and ground truth vertex"
                    "labels do not have the same shape. Skipped."
                )
                continue

            # Compute coverage mask
            covered_vertices_mask = pred_panoptic_vertex_labels != -1

            # Mask out vertices that were not covered and should not be evaluated
            covered_gt_vertex_labels = np.where(
                covered_vertices_mask, self.gt_vertex_labels, NYU40_IGNORE_LABEL
            )

            # Remove the -1 representing covered vertices from prediction
            rectified_pred_vertex_labels = np.where(
                covered_vertices_mask, pred_panoptic_vertex_labels, NYU40_IGNORE_LABEL
            )

            # Compute metrics
            method = (
                map_file_path.parent.name
                if self.experiment_type == "mapping"
                else map_file_path.parents[2].name
            )
            metrics_dict = {"Method": method, "MapID": map_file_path.stem}

            metrics_dict.update(
                panoptic_quality(
                    gt_labels=covered_gt_vertex_labels,
                    pred_labels=rectified_pred_vertex_labels,
                )
            )

            metrics_dict.update(
                mean_iou(
                    gt_labels=covered_gt_vertex_labels,
                    pred_labels=rectified_pred_vertex_labels,
                )
            )

            metrics_list.append(metrics_dict)

        metrics_df = pd.DataFrame(metrics_list).set_index("MapID").sort_index()
        metrics_file_path = self.data_dir_path / "metrics.csv"
        metrics_df.to_csv(str(metrics_file_path))


if __name__ == "__main__":
    evaluation_manager = EvaluationManager()
    evaluation_manager.evaluate()
