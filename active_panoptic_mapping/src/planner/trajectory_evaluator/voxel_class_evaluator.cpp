#include "active_panoptic_mapping/planner/trajectory_evaluator/voxel_class_evaluator.h"

#include <algorithm>
#include <vector>

namespace active_3d_planning {
namespace trajectory_evaluator {

VoxelClassEvaluator::VoxelClassEvaluator(PlannerI& planner)
    : FrontierEvaluator(planner) {}

void VoxelClassEvaluator::setupFromParamMap(Module::ParamMap* param_map) {
  FrontierEvaluator::setupFromParamMap(param_map);
  setParam<double>(param_map, "frontier_voxel_weight",
                   &p_frontier_voxel_weight_, 1.0);
  setParam<double>(param_map, "min_impact_factor", &p_min_impact_factor_, 0.0);
  setParam<double>(param_map, "new_voxel_weight", &p_new_voxel_weight_, 0.01);
  setParam<double>(param_map, "ray_angle_x", &p_ray_angle_x_, 0.0025);
  setParam<double>(param_map, "ray_angle_y", &p_ray_angle_y_, 0.0025);

  // setup map
  map_ = dynamic_cast<map::PanopticMap*>(&(planner_.getMap()));
  if (!map_) {
    planner_.printError(
        "'VoxelClassEvaluator' requires a map of type 'PanopticMap'!");
  }

  // cache voxblox constants
  c_voxel_size_ = map_->getVoxelSize();
}

bool VoxelClassEvaluator::storeTrajectoryInformation(
    TrajectorySegment* traj_in,
    const std::vector<Eigen::Vector3d>& new_voxels) {
  // Uses the default voxel info, not much gain from caching more info
  return SimulatedSensorEvaluator::storeTrajectoryInformation(traj_in,
                                                              new_voxels);
}

bool VoxelClassEvaluator::computeGainFromVisibleVoxels(
    TrajectorySegment* traj_in) {
  traj_in->gain = 0.0;
  if (!traj_in->info) {
    return false;
  }
  SimulatedSensorInfo* info =
      reinterpret_cast<SimulatedSensorInfo*>(traj_in->info.get());

  // just assume we take a single image from the last trajectory point here...
  Eigen::Vector3d origin = traj_in->trajectory.back().position_W;
  for (int i = 0; i < info->visible_voxels.size(); ++i) {
    traj_in->gain += getVoxelValue(info->visible_voxels[i], origin);
  }
  return true;
}

void VoxelClassEvaluator::visualizeTrajectoryValue(
    VisualizationMarkers* markers, const TrajectorySegment& trajectory) {
  // Display all voxels that contribute to the gain. max_impact-min_impact as
  // green-red, frontier voxels purple, unknwon voxels teal
  if (!trajectory.info) {
    return;
  }
  VisualizationMarker marker;
  marker.type = VisualizationMarker::CUBE_LIST;
  marker.scale.x() = c_voxel_size_;
  marker.scale.y() = c_voxel_size_;
  marker.scale.z() = c_voxel_size_;

  // points
  double value;
  Eigen::Vector3d origin = trajectory.trajectory.back().position_W;
  SimulatedSensorInfo* info =
      reinterpret_cast<SimulatedSensorInfo*>(trajectory.info.get());
  for (int i = 0; i < info->visible_voxels.size(); ++i) {
    value = getVoxelValue(info->visible_voxels[i], origin);
    if (value > 0.0) {
      marker.points.push_back(info->visible_voxels[i]);
      Color color;
      if (value == p_frontier_voxel_weight_) {
        color.r = 0.6;
        color.g = 0.4;
        color.b = 1.0;
        color.a = 0.5;
      } else if (value == p_new_voxel_weight_) {
        color.r = 0.0;
        color.g = 0.75;
        color.b = 1.0;
        color.a = 0.1;
      } else {
        double frac =
            (value - p_min_impact_factor_) / (1.0 - p_min_impact_factor_);
        color.r = std::min((0.5 - frac) * 2.0 + 1.0, 1.0);
        color.g = std::min((frac - 0.5) * 2.0 + 1.0, 1.0);
        color.b = 0.0;
        color.a = 0.5;
      }
      marker.colors.push_back(color);
    }
  }
  markers->addMarker(marker);

  if (p_visualize_sensor_view_) {
    sensor_model_->visualizeSensorView(markers, trajectory);
  }
}

}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
