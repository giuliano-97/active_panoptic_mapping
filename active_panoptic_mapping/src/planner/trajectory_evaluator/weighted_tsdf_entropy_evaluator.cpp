#include "active_panoptic_mapping/planner/trajectory_evaluator/weighted_tsdf_entropy_evaluator.h"

#include <cmath>

namespace active_3d_planning {
namespace trajectory_evaluator {

ModuleFactoryRegistry::Registration<WeightedTSDFEntropyEvaluator>
    WeightedTSDFEntropyEvaluator::registration("WeightedTSDFEntropyEvaluator");

WeightedTSDFEntropyEvaluator::WeightedTSDFEntropyEvaluator(PlannerI& planner)
    : VoxelClassEvaluator(planner) {}

void WeightedTSDFEntropyEvaluator::setupFromParamMap(
    Module::ParamMap* param_map) {
  VoxelClassEvaluator::setupFromParamMap(param_map);

  setParam<double>(param_map, "min_class_probability",
                   &p_min_class_probability_, 0.1);
}

double WeightedTSDFEntropyEvaluator::getVoxelValue(
    const Eigen::Vector3d& voxel, const Eigen::Vector3d& origin) {
  unsigned char voxel_state = map_->getVoxelState(voxel);
  if (voxel_state == map::TSDFMap::OCCUPIED) {
    // Surface voxel

    double probability = map_->getClassVoxelProbability(voxel);
    // If the confidence is negative only the null label has been observed,
    // which means that this might be an object of an unknown class hence there
    // is no point in trying to classify it
    if (probability < 0) {
      return 0.0;
    }

    // Compute TSDF entropy reduction
    double cur_normalized_weight =
        std::max(map_->getVoxelWeight(voxel) / map_->getMaximumWeight(), 1.0);
    double cur_tsdf_entropy =
        -cur_normalized_weight * std::log2(cur_normalized_weight);

    double z = (voxel - origin).norm();
    double spanned_angle = 2.0 * atan2(c_voxel_size_, z * 2.0);
    double new_weight = std::pow(spanned_angle, 2.0) /
                        (p_ray_angle_x_ * p_ray_angle_y_) / std::pow(z, 2.0);
    double new_normalized_weight = new_weight / map_->getMaximumWeight();
    double new_tsdf_entropy =
        -new_normalized_weight * std::log2(new_normalized_weight);

    double entropy_reduction = cur_tsdf_entropy - new_tsdf_entropy;
    double class_uncertainty_weight =
        std::sqrt(p_min_class_probability_) /
        std::sqrt(std::max(probability, p_min_class_probability_));

    double gain = entropy_reduction * class_uncertainty_weight;

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
