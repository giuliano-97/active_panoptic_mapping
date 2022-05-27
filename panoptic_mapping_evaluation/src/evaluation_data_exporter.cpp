#include "panoptic_mapping_evaluation/evaluation_data_exporter.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include <panoptic_mapping/3rd_party/csv.h>
#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/map/submap_collection.h>
#include <panoptic_mapping/tools/planning_interface.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_utils.h>

using namespace panoptic_mapping;

void EvaluationDataExporter::Config::setupParamsAndPrinting() {
  setupParam("ground_truth_pointcloud_file", &ground_truth_pointcloud_file);
  setupParam("output_suffix", &output_suffix);
  setupParam("is_single_tsdf", &is_single_tsdf);
  setupParam("export_mesh", &export_mesh);
  setupParam("refine_alignment", &refine_alignment);
  if (refine_alignment) {
    setupParam("alignment_transformation", &alignment_transformation);
  }
}

void EvaluationDataExporter::Config::checkParams() const {
  checkParamCond(std::filesystem::is_regular_file(ground_truth_pointcloud_file),
                 ground_truth_pointcloud_file + " is not a valid file!");
  checkParamCond(alignment_transformation.size() == 16,
                 "alignment_transformation must be a vector of length 16");
}

EvaluationDataExporter::EvaluationDataExporter(
    const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private_),
      config_(
          config_utilities::getConfigFromRos<EvaluationDataExporter::Config>(
              nh_private)
              .checkValid()) {
  // Initialize alignment transformation
  setupMembers();
  setupRos();
}

void EvaluationDataExporter::setupMembers() {
  gt_cloud_ptr_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(config_.ground_truth_pointcloud_file,
                                          *gt_cloud_ptr_) != 0) {
    LOG(FATAL) << "Could not load ground truth pointcloud!";
  }
  // clang-format off
  const auto& t = config_.alignment_transformation;
  default_alignment_transformation_ << t[0], t[1], t[2], t[3],
                                       t[4], t[5], t[6], t[7],
                                       t[8], t[9], t[10], t[11],
                                       t[12], t[13], t[14], t[15];
  // clang-format on
}

void EvaluationDataExporter::setupRos() {
  export_evaluation_data_server_ = nh_.advertiseService(
      "export_evaluation_data",
      &EvaluationDataExporter::exportEvaluationDataCallback, this);
}

bool EvaluationDataExporter::exportVertexLabels(
    const std::shared_ptr<panoptic_mapping::SubmapCollection>& submaps,
    const Eigen::Matrix4d& alignment_transformation,
    const std::filesystem::path& vertex_labels_file_path,
    const std::optional<std::unordered_map<int, int>>& id_to_class_map) {
  // Compute the inverse of the alignment and cast to float
  pcl::PointCloud<pcl::PointXYZ>::Ptr inv_aligned_gt_cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  Eigen::Matrix4d inv_alignment_transformation =
      alignment_transformation.inverse();
  pcl::transformPointCloud(*gt_cloud_ptr_, *inv_aligned_gt_cloud_ptr,
                           inv_alignment_transformation);

  std::vector<int> predicted_vertex_labels(
      inv_aligned_gt_cloud_ptr->points.size(), -1);
  for (size_t i = 0; i < inv_aligned_gt_cloud_ptr->points.size(); ++i) {
    const Point vertex_M(inv_aligned_gt_cloud_ptr->points[i].x,
                         inv_aligned_gt_cloud_ptr->points[i].y,
                         inv_aligned_gt_cloud_ptr->points[i].z);
    for (const Submap& submap : *submaps) {
      if (submap.isActive() && submap.hasClassLayer()) {
        // Check if the voxel the vertex is inside belongs to the submap
        // and has been observed
        const Point vertex_S = submap.getT_S_M() * vertex_M;
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
                class_block->getVoxelPtrByCoordinates(vertex_S);

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

  if (id_to_class_map) {
    for (int& label : predicted_vertex_labels) {
      if (label <= 0) {
        continue;
      }
      auto instance_class_id_pair_it =
          id_to_class_map->find(static_cast<int>(label));
      if (instance_class_id_pair_it != id_to_class_map->end()) {
        label = instance_class_id_pair_it->second * 1000 +
                instance_class_id_pair_it->first;
      } else if (label <= 40) {
        label *= 1000;
      } else {
        label = 0;
      }
    }
  }

  std::ofstream ofs(vertex_labels_file_path, std::ofstream::out);
  for (int label : predicted_vertex_labels) {
    ofs << label << "\n";
  }

  return true;
}

bool EvaluationDataExporter::exportMesh(
    const std::shared_ptr<SubmapCollection>& submaps,
    const std::filesystem::path& mesh_file_path) {
  // Collect all the meshes
  voxblox::AlignedVector<voxblox::Mesh::ConstPtr> meshes;
  for (auto& submap : *submaps) {
    auto mesh = voxblox::Mesh::Ptr(new voxblox::Mesh());
    submap.getMeshLayer().getMesh(mesh.get());
    meshes.push_back(std::const_pointer_cast<const voxblox::Mesh>(mesh));
  }

  // Merge all the meshes into one
  auto combined_mesh = voxblox::Mesh::Ptr(new voxblox::Mesh());
  voxblox::createConnectedMesh(meshes, combined_mesh.get());

  // Export the mesh as ply
  voxblox::outputMeshAsPly(mesh_file_path.string(), *combined_mesh);

  return true;
}

Eigen::Matrix4d EvaluationDataExporter::computeAlignmentTransformation(
    const std::shared_ptr<panoptic_mapping::SubmapCollection>& submaps) {
  using PointT = pcl::PointXYZ;
  using PointCloudT = pcl::PointCloud<PointT>;

  // Extract pointcloud from submaps - each mesh vertex a point
  PointCloudT::Ptr cloud_ptr(new PointCloudT());

  for (auto& submap : *submaps) {
    if (!submap.hasClassLayer()) {
      continue;
    }

    // Parse all mesh vertices.
    voxblox::BlockIndexList block_list;
    submap.getMeshLayer().getAllAllocatedMeshes(&block_list);
    for (auto& block_index : block_list) {
      auto const& mesh = submap.getMeshLayer().getMeshByIndex(block_index);
      for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        auto const& vertex = mesh.vertices.at(i);
        pcl::PointXYZ point(vertex.x(), vertex.y(), vertex.z());
        cloud_ptr->push_back(point);
      }
    }
  }

  // First transform the pointcloud with the initial alignment transformation
  PointCloudT::Ptr icp_cloud_ptr(new PointCloudT());
  pcl::transformPointCloud(*cloud_ptr, *icp_cloud_ptr,
                           default_alignment_transformation_);

  // Then refine with ICP
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setEuclideanFitnessEpsilon(1e-05);
  icp.setInputSource(icp_cloud_ptr);
  icp.setInputTarget(gt_cloud_ptr_);
  icp.align(*icp_cloud_ptr);

  // Return the final transformation
  return icp.getFinalTransformation().cast<double>() *
         default_alignment_transformation_;
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

  auto alignment_transformation = default_alignment_transformation_;
  // Refine alignment transformation using ICP
  if (config_.refine_alignment) {
    alignment_transformation = computeAlignmentTransformation(submaps);
  }

  // Export vertex labels
  std::filesystem::path vertex_labels_file_path = request.file_path;
  vertex_labels_file_path.replace_extension(".txt");

  std::filesystem::path id_to_class_map_file_path(request.file_path);
  id_to_class_map_file_path.replace_extension(".csv");
  if (std::filesystem::is_regular_file(id_to_class_map_file_path)) {
    std::unordered_map<int, int> id_to_class_map;
    io::CSVReader<2> in(id_to_class_map_file_path);
    in.read_header(io::ignore_extra_column, "InstanceID", "ClassID");
    bool read_row = true;
    while (read_row) {
      int inst = -1, cls = -1;
      read_row = in.read_row(inst, cls);
      if (inst != -1 && cls != -1) {
        id_to_class_map[inst] = cls;
      }
    }
    exportVertexLabels(submaps, alignment_transformation,
                       vertex_labels_file_path, id_to_class_map);
  } else {
    exportVertexLabels(submaps, alignment_transformation,
                       vertex_labels_file_path);
  }

  // Export mesh
  if (config_.export_mesh) {
    std::filesystem::path mesh_file_path = request.file_path;
    mesh_file_path.replace_extension(".ply");
    if (!exportMesh(submaps, mesh_file_path)) {
      LOG(ERROR) << "An error occurred while exporting the mesh! Skipped.";
    }
  }

  response.success = true;

  return true;
}