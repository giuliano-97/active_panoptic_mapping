#include <panoptic_mapping/3rd_party/config_utilities.hpp>
#include <ros/ros.h>

#include "active_panoptic_mapping_utils/evaluation_data_exporter.h"

int main(int argc, char** argv) {
  config_utilities::RequiredArguments ra(
      &argc, &argv, {"--logtostderr", "--colorlogtostderr"});

  // Setup logging.
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, false);

  // Run ros.
  ros::init(argc, argv, "evaluation_data_export_node");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  // Exporter
  EvaluationDataExporter evaluation_data_exporter(nh, nh_private);

  ros::spin();

  return 0;
}
