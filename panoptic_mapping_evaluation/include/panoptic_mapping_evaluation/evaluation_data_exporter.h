#ifndef PANOPTIC_MAPPING_EVALUATION_EVALUATION_DATA_EXPORTER_H_
#define PANOPTIC_MAPPING_EVALUATION_EVALUATION_DATA_EXPORTER_H_

#include <string>

#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/tools/planning_interface.h>
#include <panoptic_mapping_msgs/SaveLoadMap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
// clang-format off
// Must be included after ros.h to work with ros
#include <panoptic_mapping/3rd_party/config_utilities.hpp>
// clang-format on

class EvaluationDataExporter {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    std::string ground_truth_pointcloud_file = "";
    std::string output_suffix = "_eval_data.txt";
    bool is_single_tsdf = true;

    Config() { setConfigName("EvaluationDataExporter::Config"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  EvaluationDataExporter(const ros::NodeHandle& nh,
                         const ros::NodeHandle& nh_private);

  virtual ~EvaluationDataExporter() = default;

  bool exportEvaluationDataCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,
      panoptic_mapping_msgs::SaveLoadMap::Response& response);

 private:
  void setupMembers();
  void setupRos();

  void exportEvaluationData(
      const std::shared_ptr<panoptic_mapping::SubmapCollection>& submaps);

  const Config config_;
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::ServiceServer export_evaluation_data_server_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr gt_cloud_ptr_;
};

#endif  // PANOPTIC_MAPPING_EVALUATIONVALUATION_DATA_EXPORTER_H_