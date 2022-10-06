#ifndef PANOPTIC_MAPPING_EVALUATION_EVALUATION_DATA_EXPORTER_H_
#define PANOPTIC_MAPPING_EVALUATION_EVALUATION_DATA_EXPORTER_H_

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <active_panoptic_mapping_utils/ExportEvaluationData.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
// clang-format off
// Must be included after ros.h to work with ros
#include <panoptic_mapping/3rd_party/config_utilities.hpp>
// clang-format on

namespace panoptic_mapping {
class SubmapCollection;
}

class EvaluationDataExporter {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    std::string output_suffix = "_eval_data.txt";
    bool is_single_tsdf = true;
    bool refine_alignment = false;
    bool export_mesh = false;
    // clang-format off
    std::vector<double> alignment_transformation = {
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    };
    // clang-format on
    Config() { setConfigName("EvaluationDataExporter::Config"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  EvaluationDataExporter(const ros::NodeHandle& nh,
                         const ros::NodeHandle& nh_private);

  virtual ~EvaluationDataExporter() = default;

  bool exportEvaluationDataCallback(
      active_panoptic_mapping_utils::ExportEvaluationData::Request& request,
      active_panoptic_mapping_utils::ExportEvaluationData::Response& response);

 private:
  void setupMembers();
  void setupRos();

  bool exportVertexLabels(
      const std::shared_ptr<panoptic_mapping::SubmapCollection>& submaps,
      std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> gt_cloud_ptr,
      const Eigen::Matrix4d& alignment_transformation,
      const std::filesystem::path& vertex_labels_file_path,
      const std::optional<std::unordered_map<int, int>>& id_remapping =
          std::nullopt);

  bool exportMesh(
      const std::shared_ptr<panoptic_mapping::SubmapCollection>& submaps,
      const std::filesystem::path& mesh_file_path);

  Eigen::Matrix4d computeAlignmentTransformation(
      const std::shared_ptr<panoptic_mapping::SubmapCollection>& submaps,
      std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> gt_cloud_ptr);

  const Config config_;
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::ServiceServer export_evaluation_data_server_;

  Eigen::Matrix4d default_alignment_transformation_;
};

#endif  // PANOPTIC_MAPPING_EVALUATIONVALUATION_DATA_EXPORTER_H_