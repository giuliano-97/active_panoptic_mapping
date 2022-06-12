#ifndef ACTIVE_PANOPTIC_MAPPING_TRAJECTORY_EVALUATOR_PANOPTIC_UNCERTAINTY_EVALUATOR_H_
#define ACTIVE_PANOPTIC_MAPPING_TRAJECTORY_EVALUATOR_PANOPTIC_UNCERTAINTY_EVALUATOR_H_

#include <vector>

#include "active_panoptic_mapping/planner/trajectory_evaluator/voxel_class_evaluator.h"

namespace active_3d_planning {
namespace trajectory_evaluator {

class PanopticUncertaintyEvaluator : public VoxelClassEvaluator {
 public:
  explicit PanopticUncertaintyEvaluator(PlannerI& planner);  // NOLINT

  void setupFromParamMap(Module::ParamMap* param_map) override;

 protected:
  static ModuleFactoryRegistry::Registration<PanopticUncertaintyEvaluator>
      registration;

  // Override virtual function
  double getVoxelValue(const Eigen::Vector3d& voxel,
                       const Eigen::Vector3d& origin) override;

  double p_min_voxel_confidence_;
  double max_uncertainty_gain_;
};

}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
#endif  // ACTIVE_PANOPTIC_MAPPING_TRAJECTORY_EVALUATOR_PANOPTIC_UNCERTAINTY_EVALUATOR_H_
