#include "active_panoptic_mapping/planner/panoptic_map.h"

#include <utility>
#include <vector>

#include <active_3d_planning_core/data/system_constraints.h>

namespace active_3d_planning {
namespace map {

ModuleFactoryRegistry::Registration<PanopticMap> PanopticMap::registration_(
    "PanopticMap");

void PanopticMap::setupFromParamMap(Module::ParamMap* param_map) {
  setParam<float>(param_map, "check_collision_distance",
                  &p_check_collision_distance_, 0.5);
  setParam<float>(param_map, "voxel_size", &p_voxel_size_, 0.1);

  // Cache relevant data.
  c_collision_offsets_ = {Eigen::Vector3d::Zero(),   Eigen::Vector3d::UnitX(),
                          -Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
                          -Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ(),
                          -Eigen::Vector3d::UnitZ()};
  for (Eigen::Vector3d& offset : c_collision_offsets_) {
    offset *= p_check_collision_distance_;
  }
}

PanopticMap::PanopticMap(PlannerI& planner) : OccupancyMap(planner) {}

bool PanopticMap::checkIsSetup() {
  if (is_setup_) {
    return true;
  }
  if (!map_) {
    return false;
  }
  const int active_id =
      map_->getSubmapCollection().getActiveFreeSpaceSubmapID();
  if (active_id < 0) {
    return false;
  }

  active_submap_ = &map_->getSubmapCollection().getSubmap(active_id);
  is_setup_ = true;
  return true;
}

void PanopticMap::setMap(
    std::shared_ptr<const panoptic_mapping::PanopticMapper> map) {
  map_ = std::move(map);
}

bool PanopticMap::isTraversable(const Eigen::Vector3d& position,
                                const Eigen::Quaterniond& orientation) {
  if (!checkIsSetup()) {
    return false;
  }
  for (Eigen::Vector3d& offset : c_collision_offsets_) {
    const auto point = (position + offset).cast<voxblox::FloatingPoint>();
    auto block = active_submap_->getTsdfLayer().getBlockPtrByCoordinates(point);
    if (!block) {
      return false;
    }
    const panoptic_mapping::TsdfVoxel& voxel =
        block->getVoxelByCoordinates(point);
    if (voxel.distance <= 0.f || voxel.weight <= 1e-6) {
      return false;
    }
  }
  return true;
}

bool PanopticMap::isTraversablePath(
    const EigenTrajectoryPointVector& trajectory) {
  Eigen::Vector3d previous_position;
  for (auto it = trajectory.begin(); it != trajectory.end(); ++it) {
    // Only check points on a regular interval, but always include start and
    // goal point.
    if (it != trajectory.begin() && it != trajectory.end()) {
      if ((it->position_W - previous_position).norm() <=
          p_check_collision_distance_) {
        continue;
      }
    }
    if (!isTraversable(it->position_W)) {
      return false;
    }
  }
  return true;
}

bool PanopticMap::isObserved(const Eigen::Vector3d& point) {
  if (!checkIsSetup()) {
    return false;
  }
  // NOTE(schmluk): Only checks in the active map currently.
  panoptic_mapping::TsdfBlock::ConstPtr block =
      active_submap_->getTsdfLayer().getBlockPtrByCoordinates(
          point.cast<voxblox::FloatingPoint>());
  if (!block) {
    return false;
  }
  return block->getVoxelByCoordinates(point.cast<voxblox::FloatingPoint>())
             .weight > 1e-6;
}

unsigned char PanopticMap::getVoxelState(const Eigen::Vector3d& point) {
  if (!checkIsSetup()) {
    return OccupancyMap::OCCUPIED;
  }
  // Hierarchical lookup preferring the active map.
  auto position = point.cast<voxblox::FloatingPoint>();
  panoptic_mapping::TsdfBlock::ConstPtr active_block =
      active_submap_->getTsdfLayer().getBlockPtrByCoordinates(position);
  panoptic_mapping::TsdfBlock::ConstPtr past_block;

  // Look-up active map.
  if (active_block) {
    const panoptic_mapping::TsdfVoxel& voxel =
        active_block->getVoxelByCoordinates(position);
    if (voxel.weight <= 1e-6) {
      // Uknown, use past map.
    } else if (voxel.distance <= 0.f) {
      return OccupancyMap::OCCUPIED;
    } else {
      return OccupancyMap::FREE;
    }
  }
  return OccupancyMap::UNKNOWN;
}

double PanopticMap::getVoxelSize() { return p_voxel_size_; }

bool PanopticMap::getVoxelCenter(Eigen::Vector3d* center,
                                 const Eigen::Vector3d& point) {
  if (p_voxel_size_ == 0.f) {
    return false;
  }
  const float block_size =
      p_voxel_size_ * 16.f;  // Actual voxels per side don't matter since blocks
                             // are irrelevant for this function.
  voxblox::BlockIndex block_id =
      voxblox::getGridIndexFromPoint<voxblox::BlockIndex>(
          point.cast<voxblox::FloatingPoint>(), 1.f / block_size);
  *center =
      voxblox::getOriginPointFromGridIndex(block_id, block_size).cast<double>();
  voxblox::VoxelIndex voxel_id =
      voxblox::getGridIndexFromPoint<voxblox::VoxelIndex>(
          (point - *center).cast<voxblox::FloatingPoint>(),
          1.0 / p_voxel_size_);
  *center += voxblox::getCenterPointFromGridIndex(voxel_id, p_voxel_size_)
                 .cast<double>();
  return true;
}

}  // namespace map
}  // namespace active_3d_planning