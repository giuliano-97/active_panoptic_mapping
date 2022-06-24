#include "active_panoptic_mapping_core/planner/trajectory_evaluator/panoptic_uncertainty_evaluator.h"

#include <cmath>

namespace active_3d_planning {
namespace trajectory_evaluator {

ModuleFactoryRegistry::Registration<PanopticUncertaintyEvaluator>
    PanopticUncertaintyEvaluator::registration("PanopticUncertaintyEvaluator");

PanopticUncertaintyEvaluator::PanopticUncertaintyEvaluator(PlannerI& planner)
    : VoxelClassEvaluator(planner) {}

void PanopticUncertaintyEvaluator::setupFromParamMap(
    Module::ParamMap* param_map) {
  VoxelClassEvaluator::setupFromParamMap(param_map);

  setParam<double>(param_map, "min_voxel_confidence", &p_min_voxel_confidence_,
                   0.1);
  max_uncertainty_gain_ = 1 / p_min_voxel_confidence_;
}

double PanopticUncertaintyEvaluator::getVoxelValue(
    const Eigen::Vector3d& voxel, const Eigen::Vector3d& origin) {
  unsigned char voxel_state = map_->getVoxelState(voxel);
  if (voxel_state == map::TSDFMap::OCCUPIED) {
    // Surface voxel

    double confidence = map_->getClassVoxelProbability(voxel);
    // If the confidence is negative, something is wrong and we just return zero
    if (confidence < 0) {
      return 0.0;
    }

    // Basic panoptic label uncertainty gain
    double uncertainty_gain =
        (1 / std::max(p_min_voxel_confidence_, confidence)) /
        max_uncertainty_gain_;

    if (uncertainty_gain > p_min_impact_factor_) {
      return uncertainty_gain;
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
