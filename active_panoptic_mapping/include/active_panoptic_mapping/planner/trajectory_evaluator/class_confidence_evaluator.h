#ifndef ACTIVE_PANOPTIC_MAPPING_TRAJECTORY_EVALUATOR_CLASS_CONFIDENCE_EVALUATOR_H_
#define ACTIVE_PANOPTIC_MAPPING_TRAJECTORY_EVALUATOR_CLASS_CONFIDENCE_EVALUATOR_H_

#include <vector>

#include "active_panoptic_mapping/planner/trajectory_evaluator/voxel_class_evaluator.h"

namespace active_3d_planning {
namespace trajectory_evaluator {

class ClassConfidenceEvaluator : public VoxelClassEvaluator {
 public:
  explicit ClassConfidenceEvaluator(PlannerI& planner);  // NOLINT

  void setupFromParamMap(Module::ParamMap* param_map) override;

 protected:
  static ModuleFactoryRegistry::Registration<ClassConfidenceEvaluator>
      registration;

  // Override virtual function
  double getVoxelValue(const Eigen::Vector3d& voxel,
                       const Eigen::Vector3d& origin) override;

  double p_min_class_probability_;
};

}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
#endif  // ACTIVE_PANOPTIC_MAPPING_TRAJECTORY_EVALUATOR_CLASS_CONFIDENCE_EVALUATOR_H_
