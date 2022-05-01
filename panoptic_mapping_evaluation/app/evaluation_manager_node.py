from pathlib import Path

import rospy

from panoptic_mapping_msgs.srv import SaveLoadMap


class EvaluationManager:
    def __init__(self):
        rospy.init_node("evaluation_manager_node")

        self.data_dir_path = Path(rospy.get_param("~data_dir"))
        self.export_eval_data_srv_name = rospy.get_param(
            "~export_eval_data_srv_name",
            "/panoptic_mapping_evaluation/export_evaluation_data",
        )

        if not self.data_dir_path.is_dir():
            rospy.logfatal(f"{self.data_dir_path} must be a valid directory!")

        rospy.wait_for_service(self.export_eval_data_srv_name)
        self.export_eval_data_srv_proxy = rospy.ServiceProxy(
            self.export_eval_data_srv_name,
            SaveLoadMap,
        )

    def evaluate(self):
        map_files = sorted(list(self.data_dir_path.glob("**/*.panmap")))
        for map_file_path in map_files:
            rospy.loginfo(f"Exporting evaluation data for {map_file_path.name}")
            success = self.export_eval_data_srv_proxy(str(map_file_path.absolute()))
            if not success:
                rospy.logerror(f"Skipped.")


if __name__ == "__main__":
    evaluation_manager = EvaluationManager()
    evaluation_manager.evaluate()
