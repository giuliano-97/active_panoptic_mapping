#include "panoptic_mapping_evaluation/evaluation_data_exporter.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <panoptic_mapping/3rd_party/csv.h>
#include <panoptic_mapping/map/submap_collection.h>
#include <panoptic_mapping/tools/planning_interface.h>
#include <pcl/io/ply_io.h>

using namespace panoptic_mapping;

void EvaluationDataExporter::Config::setupParamsAndPrinting() {
  setupParam("ground_truth_pointcloud_file", &ground_truth_pointcloud_file);
  setupParam("output_suffix", &output_suffix);
  setupParam("is_single_tsdf", &is_single_tsdf);
}

void EvaluationDataExporter::Config::checkParams() const {
  bool gt_cloud_exists =
      std::filesystem::is_regular_file(ground_truth_pointcloud_file);
  checkParamCond(gt_cloud_exists,
                 ground_truth_pointcloud_file + " is not a valid file!");
}

EvaluationDataExporter::EvaluationDataExporter(
    const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private_),
      config_(
          config_utilities::getConfigFromRos<EvaluationDataExporter::Config>(
              nh_private)
              .checkValid()) {
  setupMembers();
  setupRos();
}

void EvaluationDataExporter::setupMembers() {
  gt_cloud_ptr_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(config_.ground_truth_pointcloud_file,
                                          *gt_cloud_ptr_) != 0) {
    LOG(FATAL) << "Could not load ground truth pointcloud!";
  }
}

void EvaluationDataExporter::setupRos() {
  export_evaluation_data_server_ = nh_.advertiseService(
      "export_evaluation_data",
      &EvaluationDataExporter::exportEvaluationDataCallback, this);
}

bool EvaluationDataExporter::exportEvaluationDataCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  // Load submap collection for which eval data should be exported
  LOG(INFO) << "Exporting evaluation data.";
  auto submaps = std::make_shared<SubmapCollection>();
  if (!submaps->loadFromFile(request.file_path)) {
    LOG(ERROR) << "Could not load panoptic map from " << request.file_path
               << ".";
    return false;
  }

  std::vector<int> predicted_vertex_labels(gt_cloud_ptr_->points.size(), -1);
  for (size_t i = 0; i < gt_cloud_ptr_->points.size(); ++i) {
    const Point vertex(gt_cloud_ptr_->points[i].x, gt_cloud_ptr_->points[i].y,
                       gt_cloud_ptr_->points[i].z);
    for (const Submap& submap : *submaps) {
      if (submap.isActive() && submap.hasClassLayer()) {
        // Check if the voxel the vertex is inside belongs to the submap
        // and has been observed
        const Point vertex_S = submap.getT_S_M() * vertex;
        if (submap.getBoundingVolume().contains_S(vertex_S)) {
          auto block_ptr =
              submap.getTsdfLayer().getBlockPtrByCoordinates(vertex_S);
          if (block_ptr) {
            const TsdfVoxel& voxel = block_ptr->getVoxelByCoordinates(vertex_S);
            // Skip if not observed
            if (voxel.weight < 1e-6) {  // TODO: harcoded parameter
              continue;
            }
          }

          auto class_block =
              submap.getClassLayer().getBlockPtrByCoordinates(vertex_S);
          if (class_block) {
            // Look up the label of the vertex
            const ClassVoxel* class_voxel =
                class_block->getVoxelPtrByCoordinates(vertex);

            predicted_vertex_labels[i] = class_voxel->getBelongingID();
          }
          // No class block but observed --> ignore label
          else {
            predicted_vertex_labels[i] = 0;
          }
        }
      }
    }
  }

  // Remap labels
  std::filesystem::path label_remapping_file_path(request.file_path);
  label_remapping_file_path.replace_extension(".csv");
  if (std::filesystem::is_regular_file(label_remapping_file_path)) {
    std::unordered_map<int, int> label_map;
    io::CSVReader<2> in(label_remapping_file_path);
    in.read_header(io::ignore_extra_column, "InstanceID", "ClassID");
    bool read_row = true;
    while (read_row) {
      int inst = -1, cls = -1;
      read_row = in.read_row(inst, cls);
      if (inst != -1 && cls != -1) {
        label_map[inst] = cls;
      }
    }

    // Now remap the instance ids to panoptic labels in the format
    // class_id * 1000 + instance_id
    for (int& label : predicted_vertex_labels) {
      if (label <= 0) {
        continue;
      }
      auto instance_class_id_pair_it = label_map.find(static_cast<int>(label));
      if (instance_class_id_pair_it != label_map.end()) {
        label = instance_class_id_pair_it->second * 1000 +
                instance_class_id_pair_it->first;
      } else {
        label *= 1000;
      }
    }
  }

  // Dump the result to disk
  std::filesystem::path eval_data_file_path = request.file_path;
  eval_data_file_path.replace_extension(".txt");
  std::ofstream ofs(eval_data_file_path, std::ofstream::out);
  for (int label : predicted_vertex_labels) {
    ofs << label << "\n";
  }

  response.success = true;

  return true;
}