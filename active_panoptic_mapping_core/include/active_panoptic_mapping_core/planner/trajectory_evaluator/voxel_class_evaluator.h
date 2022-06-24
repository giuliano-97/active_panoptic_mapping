#ifndef ACTIVE_PANOPTIC_MAPPING_CORE_TRAJECTORY_EVALUATOR_VOXEL_CLASS_EVALUATOR_H_
#define ACTIVE_PANOPTIC_MAPPING_CORE_TRAJECTORY_EVALUATOR_VOXEL_CLASS_EVALUATOR_H_

#include <vector>

#include "active_3d_planning_core/module/trajectory_evaluator/frontier_evaluator.h"
#include "active_panoptic_mapping_core/planner/panoptic_map.h"

namespace active_3d_planning {
namespace trajectory_evaluator {

// Voxel class evaluator uses the confidence of currently assigned class
// of visible surface voxels to compute the information gain
class VoxelClassEvaluator : public FrontierEvaluator {
 public:
  explicit VoxelClassEvaluator(PlannerI& planner);  // NOLINT

  // Override virtual methods
  void visualizeTrajectoryValue(VisualizationMarkers* markers,
                                const TrajectorySegment& trajectory) override;

  void setupFromParamMap(Module::ParamMap* param_map) override;

 protected:

  // Override virtual methods
  bool storeTrajectoryInformation(
      TrajectorySegment* traj_in,
      const std::vector<Eigen::Vector3d>& new_voxels) override;

  bool computeGainFromVisibleVoxels(TrajectorySegment* traj_in) override;

  // methods
  virtual double getVoxelValue(const Eigen::Vector3d& voxel,
                               const Eigen::Vector3d& origin) = 0;

  // map
  map::PanopticMap* map_;

  // params
  double p_min_impact_factor_;  // Minimum expected change, the gain is set at 0
  // here.
  double p_new_voxel_weight_;  // Multiply unobserved voxels by this weight to
  // balance quality/exploration
  double p_frontier_voxel_weight_;  // Multiply frontier voxels by this weight
  double
      p_ray_angle_x_;  // Angle [rad] spanned between 2 pixels in x direction,
  // i.e. fov/res
  double p_ray_angle_y_;

  // constants
  double c_voxel_size_;
};

}  // namespace trajectory_evaluator
}  // namespace active_3d_planning
#endif  // ACTIVE_PANOPTIC_MAPPING_CORE_TRAJECTORY_EVALUATOR_VOXEL_CLASS_EVALUATOR_H_
