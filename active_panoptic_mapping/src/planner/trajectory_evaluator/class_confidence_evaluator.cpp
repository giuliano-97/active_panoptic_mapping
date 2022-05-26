#include "active_panoptic_mapping/planner/trajectory_evaluator/class_confidence_evaluator.h"

#include <cmath>

namespace active_3d_planning {
namespace trajectory_evaluator {

ModuleFactoryRegistry::Registration<ClassConfidenceEvaluator>
    ClassConfidenceEvaluator::registration("ClassConfidenceEvaluator");

ClassConfidenceEvaluator::ClassConfidenceEvaluator(PlannerI& planner)
    : VoxelClassEvaluator(planner) {}

void ClassConfidenceEvaluator::setupFromParamMap(Module::ParamMap* param_map) {
  VoxelClassEvaluator::setupFromParamMap(param_map);

  setParam<double>(param_map, "min_class_probability",
                   &p_min_class_probability_, 0.1);
}

double ClassConfidenceEvaluator::getVoxelValue(const Eigen::Vector3d& voxel,
                                               const Eigen::Vector3d& origin) {
  unsigned char voxel_state = map_->getVoxelState(voxel);
  if (voxel_state == map::TSDFMap::OCCUPIED) {
    // Surface voxel

    double probability = map_->getClassVoxelProbability(voxel);
    // If the confidence is negative, something is wrong and we just return zero
    if (probability < 0) {
      return 0.0;
    }

    // TSDF gain
    // double z = (voxel - origin).norm();
    // double spanned_angle = 2.0 * atan2(c_voxel_size_, z * 2.0);
    // double new_weight = std::pow(spanned_angle, 2.0) /
    //                     (p_ray_angle_x_ * p_ray_angle_y_) / std::pow(z, 2.0);
    // double tsdf_gain = new_weight / (new_weight + map_->getVoxelWeight(voxel));

    // Basic semantic - probability of the voxel having a different label
    double class_gain =
        std::sqrt(p_min_class_probability_) /
        std::sqrt(std::max(probability, p_min_class_probability_));

    double gain = class_gain;
    // gain = 0.5 * gain + 0.5 * tsdf_gain;

    if (gain > p_min_impact_factor_) {
      return gain;
    }

  } else if (voxel_state == map::TSDFMap::UNKNOWN) {
    // Unobserved voxels
    if (p_frontier_voxel_weight_ > 0.0) {
      if (isFrontierVoxel(voxel)) {
        return p_frontier_voxel_weight_;
      }
    }
    return p_new_voxel_weight_;
  }
  return 0;
}

}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
